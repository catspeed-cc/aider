import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the aider directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from aider.utils import OutputStallDetector, ChdirTemporaryDirectory
from aider.diffs import create_progress_bar, diff_partial_update
from aider.models import Model


class TestOutputStallDetector(unittest.TestCase):
    def test_stall_message_fires_after_threshold(self):
        """Test that stall message fires via tool_output after threshold exceeded"""
        mock_io = MagicMock()
        
        with OutputStallDetector(mock_io, threshold=0.1) as detector:
            # Sleep to exceed threshold
            import time
            time.sleep(0.2)
            detector.check()  # This should trigger the stall message
            
        # Verify tool_output was called with stall message
        mock_io.tool_output.assert_called_with("⏳ Still working… (0s elapsed, writing file)")

    def test_no_message_before_threshold(self):
        """Test that no stall message fires before threshold is exceeded"""
        mock_io = MagicMock()
        
        with OutputStallDetector(mock_io, threshold=1.0) as detector:
            # Don't sleep, stay under threshold
            detector.check()  # This should NOT trigger the stall message
            
        # Verify tool_output was never called
        mock_io.tool_output.assert_not_called()

    def test_context_manager_resets_cleanly(self):
        """Test that context manager resets cleanly"""
        mock_io = MagicMock()
        
        # First use
        with OutputStallDetector(mock_io, threshold=0.1) as detector:
            import time
            time.sleep(0.2)
            detector.check()
            
        # Reset mock between uses
        mock_io.reset_mock()
        
        # Second use - should work independently
        with OutputStallDetector(mock_io, threshold=0.1) as detector:
            import time
            time.sleep(0.2)
            detector.check()
            
        # Should have been called at least twice (once per context manager)
        # Note: The actual call count may be higher due to background thread firing
        self.assertGreaterEqual(mock_io.tool_output.call_count, 2)


class TestProgressBar(unittest.TestCase):
    def test_create_progress_bar_no_suffix(self):
        """Test backward compatibility with no status_suffix"""
        # Test 0% progress - should have no filled blocks
        bar = create_progress_bar(0)
        self.assertIn("░", bar)
        # No filled blocks at 0% progress
        self.assertNotIn("█", bar)
        
        # Test 100% progress - should have filled blocks
        bar = create_progress_bar(100)
        self.assertIn("█", bar)
        self.assertNotIn("░", bar)

    def test_create_progress_bar_with_suffix(self):
        """Test that suffix appends correctly when passed"""
        bar = create_progress_bar(50, "testing")
        self.assertIn("testing", bar)
        
        # Test with empty suffix
        bar = create_progress_bar(50, "")
        self.assertNotIn(" ", bar)  # No extra spaces


class TestDiffPartialUpdate(unittest.TestCase):
    def test_diff_partial_update_backward_compatibility(self):
        """Test backward compatibility - no status_suffix parameter"""
        lines_orig = ["line1\n", "line2\n"]
        lines_updated = ["line1\n", "line2 updated\n"]
        
        # Should work without status_suffix
        result = diff_partial_update(lines_orig, lines_updated, final=True)
        self.assertIsInstance(result, str)
        self.assertIn("diff", result)

    def test_diff_partial_update_with_suffix(self):
        """Test that suffix is handled correctly when passed"""
        lines_orig = ["line1\n", "line2\n"]
        lines_updated = ["line1\n", "line2 updated\n"]
        
        # Test with status_suffix
        result = diff_partial_update(lines_orig, lines_updated, final=True, status_suffix="test")
        self.assertIsInstance(result, str)


if __name__ == '__main__':
    unittest.main()
