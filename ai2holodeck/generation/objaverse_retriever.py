import os

import compress_json
import compress_pickle
import numpy as np
import torch
import torch.nn.functional as F
from typing import Sequence, Tuple, List

from ai2holodeck.constants import (
    OBJATHOR_ANNOTATIONS_PATH,
    HOLODECK_THOR_ANNOTATIONS_PATH,
    OBJATHOR_FEATURES_DIR,
    HOLODECK_THOR_FEATURES_DIR,
)
from ai2holodeck.generation.utils import get_bbox_dims


class ObjathorRetriever:
    def __init__(
        self,
        clip_model,
        clip_preprocess,
        clip_tokenizer,
        sbert_model,
        retrieval_threshold,
    ):
        objathor_annotations = compress_json.load(OBJATHOR_ANNOTATIONS_PATH)
        thor_annotations = compress_json.load(HOLODECK_THOR_ANNOTATIONS_PATH)
        self.database = {**objathor_annotations, **thor_annotations}

        objathor_clip_features_dict = compress_pickle.load(
            os.path.join(OBJATHOR_FEATURES_DIR, f"clip_features.pkl")
        )  # clip features
        objathor_sbert_features_dict = compress_pickle.load(
            os.path.join(OBJATHOR_FEATURES_DIR, f"sbert_features.pkl")
        )  # sbert features
        assert (
            objathor_clip_features_dict["uids"] == objathor_sbert_features_dict["uids"]
        )

        objathor_uids = objathor_clip_features_dict["uids"]
        objathor_clip_features = objathor_clip_features_dict["img_features"].astype(
            np.float32
        )
        objathor_sbert_features = objathor_sbert_features_dict["text_features"].astype(
            np.float32
        )

        thor_clip_features_dict = compress_pickle.load(
            os.path.join(HOLODECK_THOR_FEATURES_DIR, "clip_features.pkl")
        )  # clip features
        thor_sbert_features_dict = compress_pickle.load(
            os.path.join(HOLODECK_THOR_FEATURES_DIR, "sbert_features.pkl")
        )  # clip features
        assert thor_clip_features_dict["uids"] == thor_sbert_features_dict["uids"]

        thor_uids = thor_clip_features_dict["uids"]
        thor_clip_features = thor_clip_features_dict["img_features"].astype(np.float32)
        thor_sbert_features = thor_sbert_features_dict["text_features"].astype(
            np.float32
        )

        self.clip_features = torch.from_numpy(
            np.concatenate([objathor_clip_features, thor_clip_features], axis=0)
        )
        self.clip_features = F.normalize(self.clip_features, p=2, dim=-1)

        self.sbert_features = torch.from_numpy(
            np.concatenate([objathor_sbert_features, thor_sbert_features], axis=0)
        )

        self.asset_ids = objathor_uids + thor_uids

        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer
        self.sbert_model = sbert_model

        self.retrieval_threshold = retrieval_threshold

        self.use_text = True

    def retrieve(self, queries: Sequence[str], threshold: float = 28) -> List[Tuple[str, float]]:
        """
        Retrieve 3D asset candidates for natural-language queries using CLIP (+ optional SBERT).

        For each query, the method:
        - Encodes the text with CLIP and compares it to precomputed CLIP image embeddings of every asset across
          multiple rendered views; the maximum view per asset is taken.
        - If `self.use_text` is True, adds SBERT text similarity against precomputed asset text features.

        Filtering and scoring:
        - A CLIP-only threshold is applied per (query, asset) after the max-over-views step (scores are scaled by 100).
        - Returned scores are the combined similarity: CLIP + SBERT when `self.use_text` is True, else CLIP only.

        Args:
            queries: list[str] or tuple[str, ...]; batch of one or more natural-language
                descriptions, e.g., ["a wooden dining table"]. These are tokenized and encoded as a batch.
            threshold: float; CLIP similarity cutoff on the 0–100 scaled dot-product; items below are dropped.

        Returns:
            list[tuple[str, float]]: (asset_id, score) sorted by score descending, aggregated across all queries.
            When multiple queries are provided, results are mixed (no per-query grouping or deduplication).

        Example:
            >>> retriever.retrieve(["a wooden dining table"], threshold=31)
        """
        print(f"Retrieving for queries: {queries}")

        with torch.no_grad():
            query_feature_clip = self.clip_model.encode_text(
                self.clip_tokenizer(queries)
            )

            query_feature_clip = F.normalize(query_feature_clip, p=2, dim=-1)

        # Using Einstein summation to efficiently compute the similarity between each query
        # and every view of every 3D asset.
        # The operation is a batched dot product.
        # - `query_feature_clip` has shape (i, j) where `i` is the number of queries and
        #   `j` is the feature dimension.
        # - `self.clip_features` has shape (l, k, j) where `l` is the number of assets,
        #   `k` is the number of views per asset, and `j` is the feature dimension.
        # The result `clip_similarities` will have shape (i, l, k), containing the
        # similarity score for each query `i`, asset `l`, and view `k`.
        clip_similarities = 100 * torch.einsum(
            "ij, lkj -> ilk", query_feature_clip, self.clip_features
        )

        # Take the max similarity view
        # (query, asset, view) -> (query, asset)
        clip_similarities = torch.max(clip_similarities, dim=-1).values

        query_feature_sbert = self.sbert_model.encode(
            queries, convert_to_tensor=True, show_progress_bar=False
        ) # shape: (query, feature_dim)

        # (query, feature_dim) @ (feature_dim, asset) = (query, asset)
        sbert_similarities = query_feature_sbert @ self.sbert_features.T

        if self.use_text:
            similarities = clip_similarities + sbert_similarities
        else:
            similarities = clip_similarities

        # (query, asset) -> Tuple(tensor[query_indices], tensor[asset_indices])
        threshold_indices = torch.where(clip_similarities > threshold)

        unsorted_results = []
        for query_index, asset_index in zip(*threshold_indices):
            score = similarities[query_index, asset_index].item()
            unsorted_results.append((self.asset_ids[asset_index], score))

        # sort by score, descending order
        results = sorted(unsorted_results, key=lambda x: x[1], reverse=True)

        return results


    def compute_size_difference(
        self,
        target_size: Sequence[float],
        candidates: Sequence[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Re-rank candidates by how closely their size matches a target size.

        Each candidate asset's axis-aligned bounding-box dimensions are fetched from the database, converted to
        centimeters, and sorted to be orientation-agnostic. The target size is also sorted (expected in centimeters).
        The mean absolute difference (in cm) is computed and converted to meters; the candidate's score is adjusted as:
        adjusted_score = original_score - 10 * mean_abs_difference_m.

        Args:
            target_size: 3-length sequence of numbers (x, y, z) in centimeters; ordering is ignored.
            candidates: list[tuple[str, float]]; candidates to re-rank.

        Returns:
            list[tuple[str, float]]: (asset_id, adjusted_score) sorted by adjusted_score descending.

        Notes:
            Bounding-box dimensions come from `get_bbox_dims` (stored in meters in the annotations) and are converted
            to centimeters here. This method does not mutate the input list; it returns a new list.

        Example:
            >>> retriever.compute_size_difference((160, 75, 90), candidates)
        """
        candidate_sizes = []
        for uid, _ in candidates:
            size = get_bbox_dims(self.database[uid])
            # meter -> centimeters
            size_list = [size["x"] * 100, size["y"] * 100, size["z"] * 100]
            size_list.sort() # orientation-agnostic
            candidate_sizes.append(size_list)

        candidate_sizes = torch.tensor(candidate_sizes) # shape: (num_candidates, 3)

        target_size_list = list(target_size)
        target_size_list.sort() # orientation-agnostic
        target_size = torch.tensor(target_size_list)

        size_difference = abs(candidate_sizes - target_size).mean(axis=1) / 100
        size_difference = size_difference.tolist()

        candidates_with_size_difference = []
        for i, (uid, score) in enumerate(candidates):
            candidates_with_size_difference.append(
                (uid, score - size_difference[i] * 10)
            )

        # sort the candidates by score
        candidates_with_size_difference = sorted(
            candidates_with_size_difference, key=lambda x: x[1], reverse=True
        )

        return candidates_with_size_difference
