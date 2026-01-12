# Table: disp

**Database**: financial

## Columns

| Column | Alias | Type | Description |
|--------|-------|------|-------------|
| disp_id | disposition id | integer | unique number of identifying this row of record |
| client_id |  | integer | id number of client |
| account_id |  | integer | id number of account |
| type |  | text | type of disposition |

## Business Logic & Value Descriptions

### type

- "OWNER" : "USER" : "DISPONENT"
- commonsense evidence:
- the account can only have the right to issue permanent orders or apply for loans
