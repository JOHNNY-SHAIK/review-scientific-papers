import unittest
import pandas as pd
from src.preprocessing import preprocess_dataframe

class TestPreprocessing(unittest.TestCase):
    def test_cleaning_pipeline(self):
        df = pd.DataFrame({
            "paper_id": ["p1"],
            "citation_context": ["This method significantly improves accuracy compared to prior work."],
            "label": ["Positive"]
        })
        result = preprocess_dataframe(df)
        self.assertIn("clean_text", result.columns)
        self.assertIsInstance(result.loc[0, "clean_text"], str)
        self.assertTrue(len(result.loc[0, "clean_text"]) > 0)

if __name__ == "__main__":
    unittest.main()
