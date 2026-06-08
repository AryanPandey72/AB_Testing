import unittest

from app.core.bandit.thompson import ArmState, ThompsonSamplingBandit


class ThompsonSamplingBanditTest(unittest.TestCase):
    def test_reward_updates_beta_posterior(self) -> None:
        bandit = ThompsonSamplingBandit(
            [
                ArmState(arm_id=0, arm_name="control"),
                ArmState(arm_id=1, arm_name="treatment"),
            ],
            seed=3,
        )

        bandit.update(arm_id=1, reward=1)
        bandit.update(arm_id=1, reward=0)

        arm = bandit.arms[1]
        self.assertEqual(arm.alpha, 2.0)
        self.assertEqual(arm.beta, 2.0)
        self.assertEqual(arm.pulls, 2)
        self.assertEqual(arm.rewards, 1)

    def test_choose_arm_returns_known_arm(self) -> None:
        bandit = ThompsonSamplingBandit(
            [
                ArmState(arm_id=0, arm_name="control", alpha=1, beta=50),
                ArmState(arm_id=1, arm_name="treatment", alpha=50, beta=1),
            ],
            seed=9,
        )

        decision = bandit.choose_arm()

        self.assertIn(decision.selected_arm_id, {0, 1})
        self.assertEqual(len(decision.samples), 2)


if __name__ == "__main__":
    unittest.main()

