# Table: loan

**Database**: ods

## Columns

| Column | Alias | Type | Description |
|--------|-------|------|-------------|
| loan_id |  | integer | the id number identifying the loan data |
| account_id |  | integer | the id number identifying the account |
| date |  | date | the date when the loan is approved |
| amount |  | integer | approved amount |
| duration |  | integer | loan duration |
| payments | monthly payments | real | monthly payments |
| status |  | text | repayment status |

## Business Logic & Value Descriptions

### amount

- unit：US dollar

### duration

- unit：month

### payments (monthly payments)

- unit：month

### status

- 'A' stands for contract finished, no problems;
- 'B' stands for contract finished, loan not paid;
- 'C' stands for running contract, OK so far;
- 'D' stands for running contract, client in debt
