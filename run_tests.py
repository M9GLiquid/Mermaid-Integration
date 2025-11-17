#!/usr/bin/env python3
"""
Test runner for Integration-v1 APIs

Runs all API tests and provides a summary.

Usage:
    python3 run_tests.py
"""

import subprocess
import sys
from pathlib import Path


def run_test(test_file: str, test_name: str = None) -> tuple[bool, str]:
    """
    Run a test file and return success status and output.
    
    Args:
        test_file: Path to test file
        test_name: Optional specific test to run
    
    Returns:
        Tuple of (success: bool, output: str)
    """
    integration_root = Path(__file__).parent
    test_path = integration_root / test_file
    
    if not test_path.exists():
        return False, f"Test file not found: {test_file}"
    
    try:
        cmd = [sys.executable, str(test_path)]
        if test_name:
            cmd.append(test_name)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(integration_root)
        )
        
        success = result.returncode == 0
        output = result.stdout + result.stderr
        
        return success, output
    except Exception as e:
        return False, f"Error running test: {e}"


def main():
    """Run all tests and display summary."""
    print("=" * 70)
    print("Integration-v1 API Test Suite")
    print("=" * 70)
    print()
    
    tests = [
        ("test_layout_api.py", None, "Layout API"),
        ("test_overlay_api.py", None, "Overlay API"),
    ]
    
    results = []
    
    for test_file, test_name, test_display_name in tests:
        print(f"\n{'=' * 70}")
        print(f"Running: {test_display_name}")
        print(f"{'=' * 70}\n")
        
        success, output = run_test(test_file, test_name)
        print(output)
        
        results.append((test_display_name, success))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {status}: {test_name}")
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n[SUCCESS] All API tests passed!")
        return 0
    else:
        print(f"\n[FAILURE] {failed} test suite(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
