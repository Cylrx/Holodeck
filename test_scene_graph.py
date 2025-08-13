#!/usr/bin/env python3
"""
Test script for the scene graph generation functionality.
Tests both floor and wall object constraint formats.
"""

import sys
import os

# Add the ai2holodeck path to sys.path
sys.path.insert(0, os.path.abspath('.'))

from ai2holodeck.generation.utils import to_mermaid_script, make_scene_graph

def test_floor_constraints():
    """Test floor object constraints format"""
    print("=" * 50)
    print("Testing Floor Object Constraints")
    print("=" * 50)
    
    floor_constraints = {
        'banquet_table-0': [{'type': 'global', 'constraint': 'middle'}],
        'buffet-0': [
            {'type': 'global', 'constraint': 'edge'}, 
            {'type': 'distance', 'constraint': 'far', 'target': 'banquet_table-0'}, 
            {'type': 'relative', 'constraint': 'side of', 'target': 'banquet_table-0'}
        ],
        'armchair-0': [
            {'type': 'global', 'constraint': 'edge'}, 
            {'type': 'distance', 'constraint': 'near', 'target': 'banquet_table-0'}, 
            {'type': 'direction', 'constraint': 'face to', 'target': 'banquet_table-0'}, 
            {'type': 'alignment', 'constraint': 'center aligned', 'target': 'banquet_table-0'}
        ],
        'floor_lamp-0': [
            {'type': 'global', 'constraint': 'edge'}, 
            {'type': 'distance', 'constraint': 'near', 'target': 'armchair-0'}, 
            {'type': 'relative', 'constraint': 'side of', 'target': 'armchair-0'}
        ]
    }
    
    mermaid_script = to_mermaid_script(floor_constraints)
    print("Generated Mermaid Script:")
    print(mermaid_script)
    print()
    
    # Test graph creation
    try:
        graph = make_scene_graph(floor_constraints)
        print("✓ Graph object created successfully")
    except Exception as e:
        print(f"✗ Error creating graph: {e}")
    print()

def test_wall_constraints():
    """Test wall object constraints format"""
    print("=" * 50)
    print("Testing Wall Object Constraints")
    print("=" * 50)
    
    wall_constraints = {
        'painting-0': {'target_floor_object_name': 'banquet_table-0', 'height': 160},
        'painting-1': {'target_floor_object_name': 'banquet_table-0', 'height': 160},
        'wall_shelf-0': {'target_floor_object_name': 'buffet-0', 'height': 150},
        'mirror-0': {'target_floor_object_name': 'sideboard-0', 'height': 100}
    }
    
    mermaid_script = to_mermaid_script(wall_constraints)
    print("Generated Mermaid Script:")
    print(mermaid_script)
    print()
    
    # Test graph creation
    try:
        graph = make_scene_graph(wall_constraints)
        print("✓ Graph object created successfully")
    except Exception as e:
        print(f"✗ Error creating graph: {e}")
    print()

def test_empty_constraints():
    """Test edge case with empty constraints"""
    print("=" * 50)
    print("Testing Empty Constraints")
    print("=" * 50)
    
    empty_constraints = {}
    
    mermaid_script = to_mermaid_script(empty_constraints)
    print("Generated Mermaid Script:")
    print(mermaid_script)
    print()

if __name__ == "__main__":
    test_floor_constraints()
    test_wall_constraints()
    test_empty_constraints()
    
    print("=" * 50)
    print("All tests completed!")
    print("=" * 50)
