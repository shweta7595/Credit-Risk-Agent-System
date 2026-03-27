# Credit Underwriting Policy

## Eligibility Criteria

| Parameter              | Minimum | Maximum | Notes                                   |
|------------------------|---------|---------|-----------------------------------------|
| Applicant Age          | 21      | 65      | Must be legal adult with income history |
| Annual Income          | $12,000 | —       | Must demonstrate ability to repay       |
| Employment Length      | 1 year  | —       | Stable employment preferred             |
| Credit History Length  | 2 years | —       | Longer history reduces uncertainty      |
| Debt-to-Income (DTI)  | —       | 0.43    | Total debt payments / gross income      |

## Loan Parameters

| Parameter        | Constraint                                     |
|------------------|-------------------------------------------------|
| Loan Amount      | $500 – $35,000                                  |
| Interest Rate    | Risk-based pricing; floor 5.42%, cap 23.22%     |
| Loan Grade       | A–G scale; grades E–G require additional review |
| Loan-to-Income   | Must not exceed 0.50                            |

## Decision Rules

1. **Auto-Approve** if ALL conditions met:
   - Loan grade A or B
   - DTI ≤ 0.20
   - No prior defaults on file
   - Employment length ≥ 3 years
   - Model risk score < 0.25

2. **Auto-Decline** if ANY condition met:
   - DTI > 0.50
   - Age < 21 or > 65
   - Income < $12,000
   - Model risk score > 0.85

3. **Manual Review** for all other cases:
   - Requires senior underwriter sign-off
   - Must document rationale

## Prior Default Handling

- Applicants with `cb_person_default_on_file = Y` receive a risk premium
  of +2% on the interest rate
- Prior default combined with loan grade D or worse triggers manual review
