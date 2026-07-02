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
            
        # Second use - should work independently
        with OutputStallDetector(mock_io, threshold=0.1) as detector:
            import time
            time.sleep(0.2)
            detector.check()
            
        # Should have been called twice (once per context manager)
        self.assertEqual(mock_io.tool_output.call_count, 2)


class TestProgressBar(unittest.TestCase):
    def test_create_progress_bar_no_suffix(self):
        """Test backward compatibility with no status_suffix"""
        # Test 0% progress
        bar = create_progress_bar(0)
        self.assertIn("░", bar)
        self.assertIn("█", bar)
        
        # Test 100% progress
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
        self.assertIn("test", result)


class TestOllamaContextQuery(unittest.TestCase):
    @patch('requests.get')
    def test_ollama_api_ps_with_name_field(self, mock_get):
        """Test Ollama /api/ps context query with name field in response"""
        # Mock successful API call with name field
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "test-model", "context_length": 8192}
            ]
        }
        mock_get.return_value = mock_response
        
        # Create a model instance with ollama prefix
        model = Model("ollama/test-model")
        
        # Mock io for the model
        mock_io = MagicMock()
        model.io = mock_io
        
        # Mock token_count to avoid actual token calculation
        with patch.object(model, 'token_count', return_value=1000):
            # This should use the VRAM-adjusted context from API
            kwargs = {"messages": []}
            try:
                # Call send_completion which triggers the Ollama logic
                model.send_completion(**kwargs)
            except Exception:
                pass  # We're just testing the context setting logic
            
        # Verify that tool_output was called with VRAM-adjusted context message
        mock_io.tool_output.assert_called_with("Using Ollama VRAM-adjusted context: 8192 tokens")

    @patch('requests.get')
    def test_ollama_api_ps_with_model_field(self, mock_get):
        """Test Ollama /api/ps context query with model field in response"""
        # Mock successful API call with model field
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"model": "test-model", "context_length": 4096}
            ]
        }
        mock_get.return_value = mock_response
        
        # Create a model instance with ollama prefix
        model = Model("ollama/test-model")
        
        # Mock io for the model
        mock_io = MagicMock()
        model.io = mock_io
        
        # Mock token_count to avoid actual token calculation
        with patch.object(model, 'token_count', return_value=1000):
            # This should use the VRAM-adjusted context from API
            kwargs = {"messages": []}
            try:
                # Call send_completion which triggers the Ollama logic
                model.send_completion(**kwargs)
            except Exception:
                pass  # We're just testing the context setting logic
            
        # Verify that tool_output was called with VRAM-adjusted context message
        mock_io.tool_output.assert_called_with("Using Ollama VRAM-adjusted context: 4096 tokens")

    @patch('requests.get')
    def test_ollama_api_ps_empty_models_list(self, mock_get):
        """Test Ollama /api/ps with empty models list"""
        # Mock successful API call with empty models list
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        mock_get.return_value = mock_response
        
        # Create a model instance with ollama prefix
        model = Model("ollama/test-model")
        
        # Mock io for the model
        mock_io = MagicMock()
        model.io = mock_io
        
        # Mock token_count to avoid actual token calculation
        with patch.object(model, 'token_count', return_value=1000):
            # This should fall back to heuristic calculation
            kwargs = {"messages": []}
            try:
                # Call send_completion which triggers the Ollama logic
                model.send_completion(**kwargs)
            except Exception:
                pass  # We're just testing the context setting logic
            
        # Verify that tool_warning was called with fallback message
        mock_io.tool_warning.assert_called_with("Falling back to heuristic calculation for Ollama context")

    @patch('requests.get')
    def test_ollama_api_ps_connection_error(self, mock_get):
        """Test Ollama /api/ps with ConnectionError"""
        # Mock connection error
        mock_get.side_effect = Exception("Connection failed")
        
        # Create a model instance with ollama prefix
        model = Model("ollama/test-model")
        
        # Mock io for the model
        mock_io = MagicMock()
        model.io = mock_io
        
        # Mock token_count to avoid actual token calculation
        with patch.object(model, 'token_count', return_value=1000):
            # This should fall back to heuristic calculation on error
            kwargs = {"messages": []}
            try:
                # Call send_completion which triggers the Ollama logic
                model.send_completion(**kwargs)
            except Exception:
                pass  # We're just testing the context setting logic
            
        # Verify that tool_warning was called with fallback message
        mock_io.tool_warning.assert_called_with("Falling back to heuristic calculation for Ollama context")


if __name__ == '__main__':
    unittest.main()
