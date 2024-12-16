# Party Drink Optimizer

A Python-based optimization tool that helps plan beverage purchases for parties by considering group preferences, different drink categories, and container sizes. The tool uses linear programming to minimize costs while ensuring adequate supplies based on expected consumption.

## Overview

This tool solves the common party planning challenge of determining optimal beverage quantities while considering:
- Different drink categories (e.g., spirits, beer)
- Various container sizes and their costs
- Individual preferences within the group
- Expected consumption rates
- Budget optimization

## Usage

```bash
python purchase_optimizer.py config.yaml
```

## Configuration Guide

The optimizer uses YAML configuration files to define party parameters, drink categories, and preferences. Below is a detailed explanation of each configuration section.

### Serving Units

The `serving_units` section defines the basic units of measurement for different types of drinks. Each serving unit needs:
- `name`: Identifier for the serving unit (e.g., "shot", "bottle")
- `volume_ml`: Volume in milliliters for standardization

Example:
```yaml
serving_units:
  - name: shot
    volume_ml: 44.36  # 1.5 oz shot
  - name: bottle
    volume_ml: 355    # 12 oz beer bottle
```

### Drink Categories

The `drink_categories` section defines different types of beverages and their available container sizes. Each category requires:
- `name`: Category identifier (e.g., "Spirits", "Beer")
- `serving_unit`: Reference to a defined serving unit
- `container_options`: List of available container sizes

Example:
```yaml
drink_categories:
  - name: Spirits
    serving_unit: shot
    container_options:
      - name: fifth
        volume_ml: 750
        servings: 17
      - name: handle
        volume_ml: 1750
        servings: 39
```

### Party Settings

The `party_settings` section defines event-specific parameters:
- `num_people`: Number of attendees
- `num_nights`: Duration of the event in nights
- `buffer_percentage`: Extra percentage to account for variation (e.g., 0.1 for 10% extra)
- `servings_per_person_per_night`: Expected consumption per person per night for each category

Example:
```yaml
party_settings:
  num_people: 6
  num_nights: 3
  buffer_percentage: 0.1
  servings_per_person_per_night:
    Spirits: 4    # 4 shots per person per night
    Beer: 3       # 3 beers per person per night
```

### Drink Preferences

The `drink_preferences` section lists specific beverages and their group preferences:
- `name`: Specific drink identifier
- `category`: Reference to a defined drink category
- `votes`: Preference weight from group voting
- `prices`: Container prices for each available size

Example:
```yaml
drink_preferences:
  - name: Vodka
    category: Spirits
    votes: 10
    prices:
      fifth: 20.00
      handle: 40.00
  - name: IPA
    category: Beer
    votes: 8
    prices:
      six_pack: 12.00
      case: 40.00
```

## Complete Configuration Example

Here's a full configuration example that combines all sections:

```yaml
serving_units:
  - name: shot
    volume_ml: 44.36
  - name: bottle
    volume_ml: 355

drink_categories:
  - name: Spirits
    serving_unit: shot
    container_options:
      - name: fifth
        volume_ml: 750
        servings: 17
      - name: liter
        volume_ml: 1000
        servings: 22
      - name: handle
        volume_ml: 1750
        servings: 39
  
  - name: Beer
    serving_unit: bottle
    container_options:
      - name: six_pack
        volume_ml: 2130
        servings: 6
      - name: twelve_pack
        volume_ml: 4260
        servings: 12
      - name: case
        volume_ml: 8520
        servings: 24

party_settings:
  num_people: 6
  num_nights: 3
  buffer_percentage: 0.1
  servings_per_person_per_night:
    Spirits: 4
    Beer: 3

drink_preferences:
  - name: Vodka
    category: Spirits
    votes: 10
    prices:
      fifth: 20.00
      liter: 25.00
      handle: 40.00
  
  - name: IPA
    category: Beer
    votes: 8
    prices:
      six_pack: 12.00
      twelve_pack: 22.00
      case: 40.00
```

## Output Format

The optimizer provides a detailed purchase plan showing:
- Recommended purchases by drink category
- Total servings available
- Cost breakdown by category and container size
- Total cost for the event

Example output:
```
Optimal Purchase Plan:
--------------------------------------------------

Spirits:
  Vodka:
    handle (39 shots): 2 ($80.00, 78 shots)
    Subtotal: $80.00, 78 shots

Beer:
  IPA:
    case (24 bottles): 1 ($40.00, 24 bottles)
    twelve_pack (12 bottles): 1 ($22.00, 12 bottles)
    Subtotal: $62.00, 36 bottles

Total Cost: $142.00
Spirits: 78 shots ($80.00)
Beer: 36 bottles ($62.00)
```
