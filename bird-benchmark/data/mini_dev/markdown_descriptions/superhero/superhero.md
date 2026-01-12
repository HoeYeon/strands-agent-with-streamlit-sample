# Table: superhero

**Database**: superhero

## Columns

| Column | Alias | Type | Description |
|--------|-------|------|-------------|
| id |  | integer | the unique identifier of the superhero |
| superhero_name | superhero name | text | the name of the superhero |
| full_name | full name | text | the full name of the superhero |
| gender_id | gender id | integer | the id of the superhero's gender |
| eye_colour_id | eye colour id | integer | the id of the superhero's eye color |
| hair_colour_id | hair colour id | integer | the id of the superhero's hair color |
| skin_colour_id | skin colour id | integer | the id of the superhero's skin color |
| race_id | race id | integer | the id of the superhero's race |
| publisher_id | publisher id | integer | the id of the publisher |
| alignment_id | alignment id | integer | the id of the superhero's alignment |
| height_cm | height cm | integer | the height of the superhero |
| weight_kg | weight kg | integer | the weight of the superhero |

## Business Logic & Value Descriptions

### full_name (full name)

- commonsense evidence:
- The full name of a person typically consists of their given name, also known as their first name or personal name, and their surname, also known as their last name or family name. For example, if someone's given name is "John" and their surname is "Smith," their full name would be "John Smith."

### height_cm (height cm)

- commonsense evidence:
- The unit of height is centimeter. If the height_cm is NULL or 0, it means the height of the superhero is missing.

### weight_kg (weight kg)

- commonsense evidence:
- The unit of weight is kilogram. If the weight_kg is NULL or 0, it means the weight of the superhero is missing.
