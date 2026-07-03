import unittest
from unittest.mock import patch, MagicMock, ANY
import sys
import os
import time

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
            time.sleep(0.2)
            detector.check()

        # Reset mock to isolate the second run
        mock_io.reset_mock()

        # Second use - should work independently of the first
        with OutputStallDetector(mock_io, threshold=0.1) as detector:
            time.sleep(0.2)
            detector.check()

        # Assert that the second run triggered at least once, proving it reset cleanly
        self.assertGreaterEqual(mock_io.tool_output.call_count, 1)


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


class TestOllamaContextDetection(unittest.TestCase):
    def test_ollama_context_from_api(self):
        """Test getting context from Ollama API"""
        model = Model("ollama/llama3")
        mock_io = MagicMock()
        model.io = mock_io
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "llama3",
                    "context_length": 8192
                }
            ]
        }
        
        # Patch requests.get correctly
        with patch("requests.get", return_value=mock_response) as mock_get:
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value = MagicMock()
                model.send_completion(messages=[], functions=None, stream=False)
                
                # Assert num_ctx was set correctly
                mock_completion.assert_called_once()
                kwargs = mock_completion.call_args[1]
                self.assertEqual(kwargs["num_ctx"], 8192)

    def test_ollama_context_heuristic_fallback(self):
        """Test heuristic fallback when API fails"""
        model = Model("ollama/llama3")
        mock_io = MagicMock()
        model.io = mock_io
        
        # Patch requests.get correctly
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("API Error")
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value = MagicMock()
                model.send_completion(messages=[], functions=None, stream=False)
                
                # Assert heuristic fallback was used
                mock_completion.assert_called_once()
                kwargs = mock_completion.call_args[1]
                expected_ctx = int(model.token_count([]) * 1.25) + 8192
                self.assertEqual(kwargs["num_ctx"], expected_ctx)
                mock_io.tool_warning.assert_called_with(ANY)

    def test_ollama_context_heuristic_fallback_message(self):
        """Test heuristic fallback warning message"""
        model = Model("ollama/llama3")
        mock_io = MagicMock()
        model.io = mock_io
        
        # Patch requests.get correctly
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("API Error")
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value = MagicMock()
                model.send_completion(messages=[], functions=None, stream=False)
                
                # Assert warning message starts correctly
                mock_io.tool_warning.assert_called()
                warning_msg = mock_io.tool_warning.call_args[0][0]
                self.assertTrue(warning_msg.startswith("Falling back to heuristic calculation for Ollama context:"))

    def test_ollama_context_api_base_custom(self):
        """Test custom api_base handling"""
        model = Model("ollama/llama3")
        mock_io = MagicMock()
        model.io = mock_io
        model.extra_params = {"api_base": "http://192.168.1.1:11435"}
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        
        # Patch requests.get correctly
        with patch("requests.get", return_value=mock_response) as mock_get:
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value = MagicMock()
                model.send_completion(messages=[], functions=None, stream=False)
                
                # Assert requests.get was called with correct URL
                mock_get.assert_called_once_with("http://192.168.1.1:11435/api/ps", timeout=2)

    def test_ollama_context_api_base_default(self):
        """Test default api_base handling"""
        model = Model("ollama/llama3")
        mock_io = MagicMock()
        model.io = mock_io
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        
        # Patch requests.get correctly
        with patch("requests.get", return_value=mock_response) as mock_get:
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value = MagicMock()
                model.send_completion(messages=[], functions=None, stream=False)
                
                # Assert requests.get was called with default URL
                mock_get.assert_called_once_with("http://localhost:11434/api/ps", timeout=2)

    def test_ollama_context_model_not_found(self):
        """Test handling when model not found in API response"""
        model = Model("ollama/llama3")
        mock_io = MagicMock()
        model.io = mock_io
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "mistral",
                    "context_length": 4096
                }
            ]
        }
        
        # Patch requests.get correctly
        with patch("requests.get", return_value=mock_response) as mock_get:
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value = MagicMock()
                model.send_completion(messages=[], functions=None, stream=False)
                
                # Assert heuristic fallback was used
                mock_completion.assert_called_once()
                kwargs = mock_completion.call_args[1]
                expected_ctx = int(model.token_count([]) * 1.25) + 8192
                self.assertEqual(kwargs["num_ctx"], expected_ctx)

    def test_ollama_context_empty_response(self):
        """Test handling empty API response"""
        model = Model("ollama/llama3")
        mock_io = MagicMock()
        model.io = mock_io
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        
        # Patch requests.get correctly
        with patch("requests.get", return_value=mock_response) as mock_get:
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value = MagicMock()
                model.send_completion(messages=[], functions=None, stream=False)
                
                # Assert heuristic fallback was used
                mock_completion.assert_called_once()
                kwargs = mock_completion.call_args[1]
                expected_ctx = int(model.token_count([]) * 1.25) + 8192
                self.assertEqual(kwargs["num_ctx"], expected_ctx)

    def test_ollama_context_no_details(self):
        """Test handling missing details in API response"""
        model = Model("ollama/llama3")
        mock_io = MagicMock()
        model.io = mock_io
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "llama3",
                    "context_length": 8192
                }
            ]
        }
        
        # Patch requests.get correctly
        with patch("requests.get", return_value=mock_response) as mock_get:
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value = MagicMock()
                model.send_completion(messages=[], functions=None, stream=False)
                
                # Assert num_ctx was set correctly
                mock_completion.assert_called_once()
                kwargs = mock_completion.call_args[1]
                self.assertEqual(kwargs["num_ctx"], 8192)

    def test_ollama_context_zero_context(self):
        """Test handling zero context from API"""
        model = Model("ollama/llama3")
        mock_io = MagicMock()
        model.io = mock_io
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "llama3",
                    "context_length": 0
                }
            ]
        }
        
        # Patch requests.get correctly
        with patch("requests.get", return_value=mock_response) as mock_get:
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value = MagicMock()
                model.send_completion(messages=[], functions=None, stream=False)
                
                # Assert heuristic fallback was used
                mock_completion.assert_called_once()
                kwargs = mock_completion.call_args[1]
                expected_ctx = int(model.token_count([]) * 1.25) + 8192
                self.assertEqual(kwargs["num_ctx"], expected_ctx)

    def test_ollama_context_large_context(self):
        """Test handling large context from API"""
        model = Model("ollama/llama3")
        mock_io = MagicMock()
        model.io = mock_io
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "llama3",
                    "context_length": 100000
                }
            ]
        }
        
        # Patch requests.get correctly
        with patch("requests.get", return_value=mock_response) as mock_get:
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value = MagicMock()
                model.send_completion(messages=[], functions=None, stream=False)
                
                # Assert num_ctx was set correctly
                mock_completion.assert_called_once()
                kwargs = mock_completion.call_args[1]
                self.assertEqual(kwargs["num_ctx"], 100000)
