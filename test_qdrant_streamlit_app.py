import unittest
from unittest.mock import patch, MagicMock
import streamlit as st
import requests
import json
import os
from qdrant_streamlit_app import API_URL

class TestQdrantStreamlitApp(unittest.TestCase):
    def setUp(self):
        # Mock streamlit session state
        self.mock_state = {}
        st.session_state = self.mock_state

    @patch('requests.post')
    @patch('streamlit.selectbox')
    @patch('streamlit.text_input')
    def test_ask_question_successful(self, mock_text_input, mock_selectbox, mock_post):
        # Mock the API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Test answer",
            "source_documents": [
                {"text": "Source text", "source": "test.txt", "score": 0.95}
            ]
        }
        mock_post.return_value = mock_response

        # Mock user inputs
        mock_text_input.return_value = "What is the test question?"
        mock_selectbox.return_value = "llama3-70b-8192"

        # Simulate the API call
        response = requests.post(
            f"{API_URL}/generate",
            json={
                "question": "What is the test question?",
                "collection_name": "my_collection",
                "model": "llama3-70b-8192"
            }
        )

        # Assertions
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response"], "Test answer")
        self.assertEqual(len(response.json()["source_documents"]), 1)
        self.assertEqual(response.json()["source_documents"][0]["score"], 0.95)

    @patch('requests.post')
    def test_search_documents_successful(self, mock_post):
        # Mock the API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "text": "Test result",
                    "source": "test.txt",
                    "score": 0.85
                }
            ]
        }
        mock_post.return_value = mock_response

        # Simulate the API call
        response = requests.post(
            f"{API_URL}/search",
            json={
                "query": "test query",
                "collection_name": "my_collection",
                "limit": 5
            }
        )

        # Assertions
        self.assertEqual(response.status_code, 200)
        results = response.json()["results"]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["score"], 0.85)
        self.assertEqual(results[0]["text"], "Test result")

    @patch('requests.post')
    def test_upload_document_successful(self, mock_post):
        # Mock the API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": "Document uploaded successfully"}
        mock_post.return_value = mock_response

        # Create a temporary test file
        test_file_name = "test_doc.txt"
        with open(test_file_name, "w") as f:
            f.write("Test content")

        try:
            # Simulate the API call
            response = requests.post(
                f"{API_URL}/upload_file",
                json={
                    "file_name": test_file_name,
                    "collection_name": "my_collection"
                }
            )

            # Assertions
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json()["message"],
                "Document uploaded successfully"
            )

        finally:
            # Clean up the test file
            if os.path.exists(test_file_name):
                os.remove(test_file_name)

    @patch('requests.post')
    @patch('streamlit.selectbox')
    @patch('streamlit.text_input')
    def test_ask_question_with_qwen_model(self, mock_text_input, mock_selectbox, mock_post):
        # Mock the API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Qwen model answer",
            "source_documents": [
                {"text": "Source text", "source": "test.txt", "score": 0.92}
            ]
        }
        mock_post.return_value = mock_response

        # Mock user inputs
        mock_text_input.return_value = "What is the test question?"
        mock_selectbox.return_value = "qwen-qwq-32b"

        # Simulate the API call
        response = requests.post(
            f"{API_URL}/generate",
            json={
                "question": "What is the test question?",
                "collection_name": "my_collection",
                "model": "qwen-qwq-32b"
            }
        )

        # Assertions
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response"], "Qwen model answer")
        self.assertEqual(len(response.json()["source_documents"]), 1)
        self.assertEqual(response.json()["source_documents"][0]["score"], 0.92)

if __name__ == '__main__':
    unittest.main()
