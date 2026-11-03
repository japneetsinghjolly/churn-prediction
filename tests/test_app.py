import json
import unittest

import app as churn_app


class ChurnAppTests(unittest.TestCase):
    def setUp(self):
        churn_app.app.config["TESTING"] = True
        self.client = churn_app.app.test_client()

    def sample(self, **overrides):
        data = {
            "tenure": 12,
            "MonthlyCharges": 70,
            "TotalCharges": 840,
            "InternetService": "Fiber optic",
            "Contract": "Month-to-month",
            "OnlineSecurity": "No",
        }
        data.update(overrides)
        return data

    def test_home_returns_dashboard(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Customer risk assessment", response.data)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertIn(response.status_code, (200, 503))
        self.assertIn("status", response.get_json())

    def test_encode_categories_correctly(self):
        encoded = churn_app.encode_customer(self.sample())
        values = encoded.iloc[0].to_dict()
        self.assertEqual(values["InternetService_Fiber optic"], 1.0)
        self.assertEqual(values["InternetService_No"], 0.0)
        self.assertEqual(values["Contract_One year"], 0.0)
        self.assertEqual(values["Contract_Two year"], 0.0)
        self.assertEqual(values["OnlineSecurity_Yes"], 0.0)

    def test_baseline_categories_encode_as_zeros(self):
        encoded = churn_app.encode_customer(self.sample(
            InternetService="DSL", Contract="Month-to-month", OnlineSecurity="No"
        ))
        self.assertEqual(float(encoded.iloc[0]["InternetService_Fiber optic"]), 0.0)
        self.assertEqual(float(encoded.iloc[0]["InternetService_No"]), 0.0)

    def test_invalid_numeric_value_rejected(self):
        response = self.client.post("/api/predict", json=self.sample(tenure=999))
        self.assertEqual(response.status_code, 400)
        self.assertIn("tenure", response.get_json()["error"])

    def test_invalid_category_rejected(self):
        response = self.client.post("/api/predict", json=self.sample(Contract="Lifetime"))
        self.assertEqual(response.status_code, 400)

    def test_fractional_tenure_rejected(self):
        with self.assertRaises(ValueError):
            churn_app.encode_customer(self.sample(tenure=1.5))

    def test_missing_json_rejected(self):
        response = self.client.post("/api/predict", data="not json", content_type="text/plain")
        self.assertEqual(response.status_code, 400)

    def test_prediction_response(self):
        if churn_app.model is None:
            self.skipTest("Model artifact could not be loaded in this environment")
        response = self.client.post("/api/predict", json=self.sample())
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        data = response.get_json()
        self.assertGreaterEqual(data["probability"], 0)
        self.assertLessEqual(data["probability"], 1)
        self.assertIn(data["risk"], ["Low", "Medium", "High"])
        self.assertIn("recommendation", data)


if __name__ == "__main__":
    unittest.main()
