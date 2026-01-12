# Table: Patient

**Database**: thrombosis_prediction

## Columns

| Column | Alias | Type | Description |
|--------|-------|------|-------------|
| ID |  | integer | identification of the patient |
| SEX |  | text | Sex |
| Birthday |  | date | Birthday |
| Description |  | date | the first date when a patient data was recorded |
| First Date |  | date | the date when a patient came to the hospital |
| Admission |  | text | patient was admitted to the hospital (+) or followed at the outpatient clinic (-) |
| Diagnosis |  | text | disease names |

## Business Logic & Value Descriptions

### SEX

- F: female; M: male

### Description

- null or empty: not recorded

### Admission

- patient was admitted to the hospital (+) or followed at the outpatient clinic (-)
