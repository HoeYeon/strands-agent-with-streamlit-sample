# Table: account

**Database**: ods

## Columns

| Column | Alias | Type | Description |
|--------|-------|------|-------------|
| account_id | account id | integer | the id of the account |
| district_id | location of branch | integer | location of branch |
| frequency | frequency | text | frequency of the acount |
| date | date | date | the creation date of the account |

## Business Logic & Value Descriptions

### frequency (frequency)

- "POPLATEK MESICNE" stands for monthly issuance
- "POPLATEK TYDNE" stands for weekly issuance
- "POPLATEK PO OBRATU" stands for issuance after transaction

### date (date)

- in the form YYMMDD
