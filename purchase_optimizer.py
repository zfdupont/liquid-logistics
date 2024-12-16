import pulp
import yaml
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path

@dataclass
class ServingUnit:
    """
    Represents a type of serving measurement
    For example: shots for spirits, bottles for beer
    """
    name: str
    volume_ml: float  # Store everything in ml for standardization
    
    def __str__(self):
        return self.name

@dataclass
class ContainerOption:
    """
    Represents a container size option with flexible serving units
    Examples: 
    - A fifth of spirits (750ml) contains ~17 shots
    - A 6-pack of beer contains 6 12oz bottles
    """
    name: str
    volume_ml: float
    servings: float
    serving_unit: ServingUnit
    
    @property
    def display_name(self) -> str:
        return f"{self.name} ({self.servings} {self.serving_unit}s)"

@dataclass
class DrinkCategory:
    """
    Represents a category of drink with its specific serving unit
    Examples: Spirits (measured in shots), Beer (measured in bottles)
    """
    name: str
    serving_unit: ServingUnit
    container_options: List[ContainerOption]

@dataclass
class DrinkPreference:
    """Represents preferences for a specific type of drink"""
    name: str
    category: DrinkCategory
    votes: int
    price_per_container: Dict[str, float]  # Maps container name to price

class PreferenceConfig:
    """Manages loading and validation of preference configuration from YAML"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.serving_units = self._create_serving_units()
        self.drink_categories = self._create_drink_categories()
        self._validate_config()
    
    def _load_config(self) -> dict:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_serving_units(self) -> Dict[str, ServingUnit]:
        """Create serving unit objects from config"""
        units = {}
        for unit in self.config['serving_units']:
            units[unit['name']] = ServingUnit(
                name=unit['name'],
                volume_ml=unit['volume_ml']
            )
        return units
    
    def _create_drink_categories(self) -> Dict[str, DrinkCategory]:
        """Create drink category objects from config"""
        categories = {}
        for cat in self.config['drink_categories']:
            serving_unit = self.serving_units[cat['serving_unit']]
            container_options = []
            
            for container in cat['container_options']:
                container_options.append(ContainerOption(
                    name=container['name'],
                    volume_ml=container['volume_ml'],
                    servings=container['servings'],
                    serving_unit=serving_unit
                ))
            
            categories[cat['name']] = DrinkCategory(
                name=cat['name'],
                serving_unit=serving_unit,
                container_options=container_options
            )
        return categories
    
    def _validate_config(self):
        """Validate configuration structure and required fields"""
        required_sections = {
            'serving_units', 'drink_categories', 
            'drink_preferences', 'party_settings'
        }
        if not all(section in self.config for section in required_sections):
            missing = required_sections - set(self.config.keys())
            raise ValueError(f"Missing required sections in config: {missing}")
        
        # Validate drink preferences
        for drink in self.config['drink_preferences']:
            if not all(key in drink for key in ['name', 'category', 'votes', 'prices']):
                raise ValueError(f"Invalid drink preference configuration: {drink}")
            
            category = self.drink_categories[drink['category']]
            container_names = {c.name for c in category.container_options}
            price_names = set(drink['prices'].keys())
            
            if container_names != price_names:
                raise ValueError(
                    f"Mismatch in container sizes and prices for {drink['name']}: "
                    f"missing {container_names - price_names}"
                )
    
    def get_drink_preferences(self) -> List[DrinkPreference]:
        """Convert drink preferences from config to DrinkPreference objects"""
        preferences = []
        for drink in self.config['drink_preferences']:
            category = self.drink_categories[drink['category']]
            preferences.append(DrinkPreference(
                name=drink['name'],
                category=category,
                votes=drink['votes'],
                price_per_container=drink['prices']
            ))
        return preferences
    
    def get_party_settings(self) -> Dict[str, float]:
        """Get party settings from config"""
        return self.config['party_settings']

def optimize_alcohol_purchases(
    preferences: List[DrinkPreference],
    num_people: int,
    servings_per_person_per_night: Dict[str, float],
    num_nights: int,
    buffer_percentage: float = 0.1
) -> Dict[Tuple[str, str], int]:
    """
    Optimize alcohol purchases considering different drink categories
    
    Args:
        preferences: List of drink preferences
        num_people: Number of people
        servings_per_person_per_night: Dict mapping category name to servings per person
        num_nights: Number of nights
        buffer_percentage: Extra buffer percentage
    """
    # Calculate total votes per category
    category_votes = {}
    for pref in preferences:
        cat_name = pref.category.name
        if cat_name not in category_votes:
            category_votes[cat_name] = 0
        category_votes[cat_name] += pref.votes
    
    # Create optimization problem
    prob = pulp.LpProblem("Alcohol_Optimization", pulp.LpMinimize)
    
    # Decision variables
    purchase_vars = {}
    for pref in preferences:
        for container in pref.category.container_options:
            var_name = f"{pref.name}_{container.name}"
            purchase_vars[var_name] = pulp.LpVariable(var_name, 0, None, pulp.LpInteger)
    
    # Objective function: minimize total cost
    prob += pulp.lpSum(
        purchase_vars[f"{pref.name}_{container.name}"] * 
        pref.price_per_container[container.name]
        for pref in preferences
        for container in pref.category.container_options
    )
    
    # Constraints per category
    for pref in preferences:
        cat_name = pref.category.name
        category_ratio = pref.votes / category_votes[cat_name]
        
        # Calculate total servings needed for this drink
        total_servings = (
            num_people * 
            servings_per_person_per_night[cat_name] * 
            num_nights * 
            (1 + buffer_percentage) * 
            category_ratio
        )
        
        # Ensure enough servings are purchased
        prob += pulp.lpSum(
            purchase_vars[f"{pref.name}_{container.name}"] * container.servings
            for container in pref.category.container_options
        ) >= total_servings
    
    # Solve the problem
    prob.solve()
    
    # Extract results
    results = {}
    for pref in preferences:
        for container in pref.category.container_options:
            var_name = f"{pref.name}_{container.name}"
            quantity = int(purchase_vars[var_name].value())
            if quantity > 0:
                results[(pref.name, container.name)] = quantity
    
    return results

def print_purchase_plan(
    results: Dict[Tuple[str, str], int],
    preferences: List[DrinkPreference]
):
    """Pretty print the purchase plan with costs and totals by category"""
    total_cost = 0
    category_totals = {}
    
    print("\nOptimal Purchase Plan:")
    print("-" * 50)
    
    # Group preferences by category
    by_category = {}
    for pref in preferences:
        if pref.category.name not in by_category:
            by_category[pref.category.name] = []
        by_category[pref.category.name].append(pref)
    
    # Print results by category
    for category_name, category_prefs in by_category.items():
        print(f"\n{category_name}:")
        category_cost = 0
        category_servings = 0
        
        for pref in category_prefs:
            drink_cost = 0
            drink_servings = 0
            print(f"\n  {pref.name}:")
            
            for container in pref.category.container_options:
                quantity = results.get((pref.name, container.name), 0)
                if quantity > 0:
                    cost = quantity * pref.price_per_container[container.name]
                    servings = quantity * container.servings
                    
                    print(f"    {container.display_name}: {quantity} "
                          f"(${cost:.2f}, {servings} {pref.category.serving_unit}s)")
                    
                    drink_cost += cost
                    drink_servings += servings
            
            if drink_cost > 0:
                print(f"    Subtotal: ${drink_cost:.2f}, "
                      f"{drink_servings} {pref.category.serving_unit}s")
            
            category_cost += drink_cost
            category_servings += drink_servings
        
        if category_cost > 0:
            print(f"\n  Category Total: ${category_cost:.2f}, "
                  f"{category_servings} {category_prefs[0].category.serving_unit}s")
            category_totals[category_name] = (category_cost, category_servings)
            total_cost += category_cost
    
    print("\n" + "=" * 50)
    print(f"Total Cost: ${total_cost:.2f}")
    for category, (cost, servings) in category_totals.items():
        print(f"{category}: {servings} servings (${cost:.2f})")

def main(config_path: str):
    """Main function to run optimization from config file"""
    config = PreferenceConfig(config_path)
    preferences = config.get_drink_preferences()
    settings = config.get_party_settings()
    
    results = optimize_alcohol_purchases(
        preferences=preferences,
        num_people=settings['num_people'],
        servings_per_person_per_night=settings['servings_per_person_per_night'],
        num_nights=settings['num_nights'],
        buffer_percentage=settings['buffer_percentage']
    )
    
    print_purchase_plan(results, preferences)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <config_path>")
        sys.exit(1)
    main(sys.argv[1])