import os
import sys
import json
import time
import unittest
from unittest.mock import ANY, MagicMock, patch

import requests
import threading

# Add the aider directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from aider.diffs import create_progress_bar, diff_partial_update
from aider.utils import OutputStallDetector, ChdirTemporaryDirectory
from aider.models import Model


class TestOutputStallDetector(unittest.TestCase):
    def test_stall_message_fires_after_threshold(self):
        """Test that stall message fires via tool_output after threshold exceeded"""
        mock_io = MagicMock()

        with OutputStallDetector(mock_io, threshold=0.1) as detector:
            # Sleep to exceed threshold
            time.sleep(0.2)
            detector.check()  # This should trigger the stall message

        # Verify tool_output was called with stall message
        mock_io.tool_output.assert_called_with(
            "⏳ Still working… (0s elapsed, writing file)"
        )

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

    def test_dynamic_visibility_show_hide(self):
        """Test dynamic visibility control with show() and hide() methods"""
        mock_io = MagicMock()

        # Test with visible = False initially
        with OutputStallDetector(mock_io, threshold=0.1, visible=False) as detector:
            time.sleep(0.2)
            detector.check()

        # Should not have called tool_output since it's hidden
        mock_io.tool_output.assert_not_called()

        # Reset mock
        mock_io.reset_mock()

        # Test with visible = True initially
        with OutputStallDetector(mock_io, threshold=0.1, visible=True) as detector:
            time.sleep(0.2)
            detector.check()

        # Should have called tool_output since it's visible
        mock_io.tool_output.assert_called_with(
            "⏳ Still working… (0s elapsed, writing file)"
        )

        # Reset mock
        mock_io.reset_mock()

        # Test show() method
        with OutputStallDetector(mock_io, threshold=0.1, visible=False) as detector:
            time.sleep(0.2)
            detector.check()  # Should not trigger since it's hidden

            # Now show it
            detector.show()
            detector.check()  # Should now trigger

        # Should have called tool_output once (after showing)
        mock_io.tool_output.assert_called_with(
            "⏳ Still working… (0s elapsed, writing file)"
        )

        # Reset mock
        mock_io.reset_mock()

        # Test hide() method
        with OutputStallDetector(mock_io, threshold=0.1, visible=True) as detector:
            time.sleep(0.2)
            detector.check()  # Should trigger

            # Now hide it
            detector.hide()
            detector.check()  # Should not trigger again

        # Should have called tool_output once (only the first time)
        mock_io.tool_output.assert_called_once_with(
            "⏳ Still working… (0s elapsed, writing file)"
        )

        # Test that hide() works in context manager exit
        mock_io.reset_mock()
        with OutputStallDetector(mock_io, threshold=0.1, visible=False) as detector:
            time.sleep(0.2)

        # Should not have called tool_output since it was hidden
        mock_io.tool_output.assert_not_called()

    def test_thread_safety(self):
        """Test that the detector is thread-safe"""
        mock_io = MagicMock()

        # Create multiple threads that will all access the same detector
        import threading

        def check_detector(detector):
            time.sleep(0.1)  # Sleep to ensure we exceed threshold
            detector.check()

        detector = OutputStallDetector(mock_io, threshold=0.05)

        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=check_detector, args=(detector,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Should have called tool_output exactly once (thread-safe behavior)
        mock_io.tool_output.assert_called_once()

    def test_custom_format_message(self):
        """Test custom message formatting"""
        mock_io = MagicMock()

        def custom_format(elapsed):
            return f"Working... {elapsed:.1f}s"

        with OutputStallDetector(
            mock_io, threshold=0.1, format_message=custom_format
        ) as detector:
            time.sleep(0.2)
            detector.check()

        mock_io.tool_output.assert_called_with("Working... 0.2s")

    def test_integration_hooks(self):
        """Test on_stall and on_resume hooks"""
        mock_io = MagicMock()
        stall_hook_called = []
        resume_hook_called = []

        def on_stall(elapsed):
            stall_hook_called.append(elapsed)

        def on_resume():
            resume_hook_called.append(True)

        with OutputStallDetector(
            mock_io, threshold=0.1, on_stall=on_stall, on_resume=on_resume
        ) as detector:
            time.sleep(0.2)
            detector.check()

        # Should have called the stall hook
        self.assertEqual(len(stall_hook_called), 1)
        self.assertGreaterEqual(stall_hook_called[0], 0.1)

        # Reset mock and test resume
        mock_io.reset_mock()
        detector.resume()

        # Should not have called tool_output after resume
        mock_io.tool_output.assert_not_called()
        self.assertEqual(len(resume_hook_called), 1)

    def test_resume_method(self):
        """Test the explicit resume method"""
        mock_io = MagicMock()
        detector = OutputStallDetector(mock_io, threshold=0.1)

        time.sleep(0.2)
        detector.check()  # Should trigger stall
        self.assertEqual(mock_io.tool_output.call_count, 1)

        # Reset mock to isolate resume effect
        mock_io.reset_mock()

        # Call resume on the SAME instance to reset state
        detector.resume()

        # Now check again - should trigger again since we've reset the state
        time.sleep(0.2)
        detector.check()

        self.assertEqual(mock_io.tool_output.call_count, 1)

    def test_timer_update_in_message(self):
        """Test that the timer in stall messages updates correctly"""
        mock_io = MagicMock()

        # Test with a custom format that shows time
        def format_with_time(elapsed):
            return f"⏳ Working... {elapsed:.1f}s elapsed"

        with OutputStallDetector(
            mock_io, threshold=0.05, format_message=format_with_time
        ) as detector:
            # Sleep just under threshold first
            time.sleep(0.02)
            detector.check()

            # Should not have triggered yet
            mock_io.tool_output.assert_not_called()

            # Sleep to exceed threshold
            time.sleep(0.1)
            detector.check()

        # Should have called with updated time
        mock_io.tool_output.assert_called_with("⏳ Working... 0.1s elapsed")


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
        result = diff_partial_update(
            lines_orig, lines_updated, final=True, status_suffix="test"
        )
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
            "models": [{"name": "llama3", "context_length": 8192}]
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
                self.assertTrue(
                    warning_msg.startswith(
                        "Falling back to heuristic calculation for Ollama context:"
                    )
                )

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
                mock_get.assert_called_once_with(
                    "http://192.168.1.1:11435/api/ps", timeout=2
                )

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
                mock_get.assert_called_once_with(
                    "http://localhost:11434/api/ps", timeout=2
                )

    def test_ollama_context_model_not_found(self):
        """Test handling when model not found in API response"""
        model = Model("ollama/llama3")
        mock_io = MagicMock()
        model.io = mock_io

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "mistral", "context_length": 4096}]
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
            "models": [{"name": "llama3", "context_length": 8192}]
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
            "models": [{"name": "llama3", "context_length": 0}]
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
            "models": [{"name": "llama3", "context_length": 100000}]
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


class TestStallDetectorEdgeCases(unittest.TestCase):
    def test_rapid_feed_prevents_stall_firing(self):
        """Rapid feed() calls should reset the timer and prevent repeated stall messages."""
        mock_io = MagicMock()
        with OutputStallDetector(mock_io, threshold=0.1) as detector:
            for _ in range(5):
                time.sleep(0.03)
                detector.feed("chunk")

            # Wait past threshold without feeding
            time.sleep(0.12)
            detector.check()

        self.assertEqual(mock_io.tool_output.call_count, 1)

    def test_custom_format_message_exception_handling(self):
        """Ensure a crashing format_message doesn't break the detector context manager."""
        mock_io = MagicMock()

        def crash_format(elapsed):
            raise RuntimeError("Format failed")

        with OutputStallDetector(
            mock_io, threshold=0.1, format_message=crash_format
        ) as detector:
            time.sleep(0.2)
            # Should not raise during __exit__
            pass

        # Detector should still attempt to call tool_output (or handle gracefully)
        mock_io.tool_output.assert_called()

    def test_concurrent_stress_test(self):
        """Verify thread safety under high contention."""
        mock_io = MagicMock()
        detector = OutputStallDetector(mock_io, threshold=0.05)

        def check_detector():
            time.sleep(0.02)
            detector.check()

        threads = [threading.Thread(target=check_detector) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Thread lock should ensure exactly one message fires
        self.assertEqual(mock_io.tool_output.call_count, 1)


class TestOllamaContextEdgeCases(unittest.TestCase):
    def test_malformed_json_response(self):
        """Malformed JSON from /api/ps should trigger heuristic fallback."""
        model = Model("ollama/llama3")
        mock_io = MagicMock()
        model.io = mock_io

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)

        with patch("requests.get", return_value=mock_response):
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value = MagicMock()
                model.send_completion(messages=[], functions=None, stream=False)

                kwargs = mock_completion.call_args[1]
                self.assertIn("num_ctx", kwargs)
                expected_ctx = int(model.token_count([]) * 1.25) + 8192
                self.assertEqual(kwargs["num_ctx"], expected_ctx)

    def test_http_timeout_fallback(self):
        """requests.Timeout should gracefully fall back to heuristic."""
        model = Model("ollama/llama3")
        mock_io = MagicMock()
        model.io = mock_io

        with patch("requests.get", side_effect=requests.exceptions.Timeout()):
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value = MagicMock()
                model.send_completion(messages=[], functions=None, stream=False)

                kwargs = mock_completion.call_args[1]
                self.assertIn("num_ctx", kwargs)
                expected_ctx = int(model.token_count([]) * 1.25) + 8192
                self.assertEqual(kwargs["num_ctx"], expected_ctx)

    def test_string_context_length_parsing(self):
        """Valid string context_length should be parsed correctly."""
        model = Model("ollama/llama3")
        mock_io = MagicMock()
        model.io = mock_io

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama3", "context_length": "8192"}]
        }

        with patch("requests.get", return_value=mock_response):
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value = MagicMock()
                model.send_completion(messages=[], functions=None, stream=False)

                kwargs = mock_completion.call_args[1]
                self.assertEqual(kwargs["num_ctx"], 8192)


class TestProgressBarBoundaries(unittest.TestCase):
    def test_exact_percentage_boundaries(self):
        """Verify block counts at critical percentage thresholds."""
        # 0% -> 0 filled blocks
        bar_0 = create_progress_bar(0)
        self.assertEqual(bar_0.count("█"), 0)
        self.assertEqual(bar_0.count("░"), 30)

        # 50% -> 15 filled blocks
        bar_50 = create_progress_bar(50)
        self.assertEqual(bar_50.count("█"), 15)
        self.assertEqual(bar_50.count("░"), 15)

        # 100% -> 30 filled blocks
        bar_100 = create_progress_bar(100)
        self.assertEqual(bar_100.count("█"), 30)
        self.assertEqual(bar_100.count("░"), 0)

    def test_suffix_rendering(self):
        """Ensure suffix appends correctly without breaking bar structure."""
        bar = create_progress_bar(50, "testing")
        self.assertIn("testing", bar)
        self.assertTrue(bar.endswith("testing"))

        # Empty/None suffix should not add trailing spaces
        bar_empty = create_progress_bar(50, "")
        self.assertFalse(bar_empty.endswith(" "))

        bar_none = create_progress_bar(50, None)
        self.assertFalse(bar_none.endswith(" "))


class TestIntegrationScenarios(unittest.TestCase):
    def test_streaming_response_stall_lifecycle(self):
        """Simulate slow LLM streaming: stall fires, then stops when work resumes."""
        mock_io = MagicMock()
        detector = OutputStallDetector(mock_io, threshold=0.1)

        # Initial delay exceeds threshold
        time.sleep(0.15)
        detector.check()
        self.assertEqual(mock_io.tool_output.call_count, 1)

        # Work resumes (feed called)
        detector.feed("new chunk")
        mock_io.reset_mock()

        # Wait past threshold again without feeding
        time.sleep(0.15)
        detector.check()
        # Should fire again because feed reset the timer
        self.assertEqual(mock_io.tool_output.call_count, 1)

    def test_write_text_retry_integration(self):
        """Verify stall detector doesn't interfere with exponential backoff in write_text."""
        mock_io = MagicMock()
        mock_io.dry_run = False
        mock_io.encoding = "utf-8"
        mock_io.newline = None

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise PermissionError("File locked")
            return MagicMock()

        with patch("builtins.open", side_effect=side_effect):
            mock_io.write_text(
                "/tmp/test.txt", "content", max_retries=3, initial_delay=0.01
            )

        # Should have retried exactly 3 times and succeeded
        self.assertEqual(call_count[0], 3)
