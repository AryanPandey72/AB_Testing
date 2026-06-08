import unittest

import pandas as pd

from app.core.causal.estimators import CausalConfig, estimate_treatment_effects
from app.core.data.criteo_schema import FEATURE_COLUMNS


class CausalEstimatorTest(unittest.TestCase):
    def test_ipw_returns_expected_shape(self) -> None:
        rows = []
        for i in range(120):
            x = i / 120
            treatment = 1 if i % 3 else 0
            conversion = 1 if treatment == 1 and i % 4 == 0 else 0
            row = {feature: x + j * 0.01 for j, feature in enumerate(FEATURE_COLUMNS)}
            row.update(
                {
                    "treatment": treatment,
                    "conversion": conversion,
                    "visit": conversion,
                    "exposure": treatment,
                }
            )
            rows.append(row)

        result = estimate_treatment_effects(
            pd.DataFrame(rows),
            CausalConfig(outcome_col="conversion", treatment_col="treatment"),
        )

        self.assertEqual(result["status"] if "status" in result else "ok", "ok")
        self.assertIn("ipw_ate", result)
        self.assertIn("psm_ate", result)
        self.assertIn("covariate_balance", result)


if __name__ == "__main__":
    unittest.main()

