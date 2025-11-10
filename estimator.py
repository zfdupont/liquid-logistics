import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd

@dataclass
class Person:
    """Represents a person with their drinking characteristics"""
    name: str
    weight_lbs: float
    sex: str  # 'M' or 'F'
    max_BAC: float  # Self-imposed BAC limit (e.g., 0.08, 0.10, 0.12)
    base_drinking_tendency: float  # 0.0-1.0, how likely they are to drink when opportunity arises
    
    @property
    def r_value(self) -> float:
        """Widmark r-value: volume of distribution"""
        return 0.68 if self.sex == 'M' else 0.55
    
    @property
    def metabolism_rate(self) -> float:
        """BAC decrease per hour (typically 0.015-0.018)"""
        return 0.015


class DrinkingSimulator:
    """Monte Carlo simulator for drinking behavior over a time period"""
    
    # Standard drink parameters
    STANDARD_DRINK_OZ = 1.5  # Shot size
    ALCOHOL_CONTENT = 0.4  # 40% alcohol (80 proof)
    PURE_ALCOHOL_PER_DRINK = STANDARD_DRINK_OZ * ALCOHOL_CONTENT  # 0.6 oz
    
    def __init__(self, person: Person, session_hours: float = 8, 
                 decision_interval_minutes: float = 25):
        """
        Initialize simulator
        
        Args:
            person: Person object with characteristics
            session_hours: Length of drinking session (e.g., 6pm-2am = 8 hours)
            decision_interval_minutes: Time between drinking decisions
        """
        self.person = person
        self.session_hours = session_hours
        self.decision_interval_minutes = decision_interval_minutes
    
    def calculate_BAC_increase(self, drinks: int = 1) -> float:
        """Calculate BAC increase from consuming drinks using Widmark formula"""
        total_pure_alcohol = self.PURE_ALCOHOL_PER_DRINK * drinks
        # Widmark: BAC = (A * 5.14) / (W * r)
        bac_increase = (total_pure_alcohol * 5.14) / (self.person.weight_lbs * self.person.r_value)
        return bac_increase
    
    def calculate_drinking_probability(self, current_BAC: float) -> float:
        """
        Calculate probability of drinking based on current BAC
        
        Uses a sigmoid-like function that decreases as BAC approaches max
        """
        if current_BAC >= self.person.max_BAC:
            return 0.0
        
        # Base probability modified by how close to max BAC
        bac_ratio = current_BAC / self.person.max_BAC
        
        if bac_ratio < 0.3:  # Low BAC: high probability
            base_prob = 0.8
        elif bac_ratio < 0.6:  # Medium BAC: moderate probability
            base_prob = 0.6
        elif bac_ratio < 0.85:  # Getting high: lower probability
            base_prob = 0.35
        else:  # Near max: very low probability
            base_prob = 0.15
        
        # Modulate by person's base drinking tendency
        probability = base_prob * self.person.base_drinking_tendency
        
        # Apply dampening as approaching max BAC
        dampening = 1 - (bac_ratio ** 2)
        
        return probability * dampening
    
    def simulate_single_session(self, verbose: bool = False) -> Tuple[int, List[float], List[float]]:
        """
        Simulate a single drinking session
        
        Returns:
            (total_drinks, bac_history, time_history)
        """
        current_BAC = 0.0
        drinks_consumed = 0
        time_minutes = 0
        
        bac_history = [0.0]
        time_history = [0.0]
        
        while time_minutes < self.session_hours * 60:
            # Metabolism reduces BAC over time interval
            time_elapsed_hours = self.decision_interval_minutes / 60
            bac_decrease = self.person.metabolism_rate * time_elapsed_hours
            current_BAC = max(0, current_BAC - bac_decrease)
            
            # Calculate probability of drinking
            drink_prob = self.calculate_drinking_probability(current_BAC)
            
            # Make drinking decision
            if np.random.random() < drink_prob:
                drinks_consumed += 1
                current_BAC += self.calculate_BAC_increase()
                
                if verbose:
                    print(f"Time: {time_minutes/60:.1f}h - Drink #{drinks_consumed}, "
                          f"BAC: {current_BAC:.3f}")
            
            time_minutes += self.decision_interval_minutes
            bac_history.append(current_BAC)
            time_history.append(time_minutes / 60)
        
        return drinks_consumed, bac_history, time_history
    
    def run_simulations(self, n_simulations: int = 1000, 
                       verbose: bool = False) -> np.ndarray:
        """
        Run multiple Monte Carlo simulations
        
        Returns:
            Array of total drinks consumed in each simulation
        """
        results = []
        
        for i in range(n_simulations):
            drinks, _, _ = self.simulate_single_session(verbose=verbose and i == 0)
            results.append(drinks)
        
        return np.array(results)
    
    def analyze_results(self, results: np.ndarray) -> dict:
        """Analyze simulation results and return statistics"""
        return {
            'mean': np.mean(results),
            'median': np.median(results),
            'std': np.std(results),
            'min': np.min(results),
            'max': np.max(results),
            'percentile_25': np.percentile(results, 25),
            'percentile_75': np.percentile(results, 75),
            'percentile_90': np.percentile(results, 90),
            'percentile_95': np.percentile(results, 95)
        }
    
    def plot_results(self, results: np.ndarray, save_path: str = None):
        """Create visualization of simulation results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histogram of drinks consumed
        axes[0, 0].hist(results, bins=range(int(results.min()), int(results.max()) + 2), 
                       edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.mean(results), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(results):.1f}')
        axes[0, 0].axvline(np.median(results), color='green', linestyle='--', 
                          label=f'Median: {np.median(results):.1f}')
        axes[0, 0].set_xlabel('Total Drinks Consumed')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'Distribution of Drinks Consumed ({len(results)} simulations)')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Calculate drinks per hour for each simulation
        drinking_rates = results / self.session_hours
        axes[0, 1].hist(drinking_rates, bins=20, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Drinks per Hour')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Drinking Rate')
        axes[0, 1].axvline(np.mean(drinking_rates), color='red', linestyle='--', 
            label=f'Mean: {np.mean(drinking_rates):.2f} drinks/hr')
        
        # Sample BAC trajectory
        _, bac_history, time_history = self.simulate_single_session()
        axes[1, 0].plot(time_history, bac_history, linewidth=2)
        axes[1, 0].axhline(self.person.max_BAC, color='red', linestyle='--', 
                          label=f'Max BAC: {self.person.max_BAC:.2f}')
        axes[1, 0].axhline(0.08, color='orange', linestyle=':', 
                          label='Legal limit: 0.08')
        axes[1, 0].set_xlabel('Time (hours)')
        axes[1, 0].set_ylabel('BAC')
        axes[1, 0].set_title('Sample BAC Trajectory')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Cumulative distribution
        sorted_results = np.sort(results)
        cumulative = np.arange(1, len(sorted_results) + 1) / len(sorted_results)
        axes[1, 1].plot(sorted_results, cumulative, linewidth=2)
        axes[1, 1].axhline(0.75, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].axvline(np.percentile(results, 75), color='red', 
                          linestyle='--', alpha=0.5, 
                          label=f'75th percentile: {np.percentile(results, 75):.1f}')
        axes[1, 1].set_xlabel('Total Drinks')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Distribution Function')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class GroupSimulator:
    """Simulate drinking for a group of people"""
    
    def __init__(self, people: List[Person], session_hours: float = 8,
                 decision_interval_minutes: float = 25):
        self.people = people
        self.session_hours = session_hours
        self.decision_interval_minutes = decision_interval_minutes
    
    def simulate_group(self, n_simulations: int = 1000) -> pd.DataFrame:
        """
        Simulate drinking for entire group
        
        Returns:
            DataFrame with results for each person
        """
        results = []
        
        for person in self.people:
            simulator = DrinkingSimulator(
                person, 
                self.session_hours, 
                self.decision_interval_minutes
            )
            
            person_results = simulator.run_simulations(n_simulations)
            stats = simulator.analyze_results(person_results)
            
            results.append({
                'name': person.name,
                'weight': person.weight_lbs,
                'sex': person.sex,
                'max_BAC': person.max_BAC,
                'tendency': person.base_drinking_tendency,
                'mean_drinks': stats['mean'],
                'median_drinks': stats['median'],
                'std_drinks': stats['std'],
                'p75_drinks': stats['percentile_75'],
                'p90_drinks': stats['percentile_90'],
                'p95_drinks': stats['percentile_95']
            })
        
        return pd.DataFrame(results)
    
    def estimate_total_drinks(self, n_simulations: int = 1000, 
                            percentile: int = 75) -> dict:
        """
        Estimate total drinks needed for group
        
        Args:
            percentile: Use this percentile for conservative estimate (75 or 90 recommended)
        
        Returns:
            Dictionary with total estimates
        """
        df = self.simulate_group(n_simulations)
        
        if percentile == 75:
            total = df['p75_drinks'].sum()
        elif percentile == 90:
            total = df['p90_drinks'].sum()
        elif percentile == 95:
            total = df['p95_drinks'].sum()
        else:
            total = df['mean_drinks'].sum()
        
        return {
            'total_drinks': total,
            'per_person_breakdown': df[['name', f'p{percentile}_drinks']].to_dict('records')
        }


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("=" * 60)
    print("MONTE CARLO DRINKING SIMULATION")
    print("=" * 60)
    
    # Example 1: Single person simulation
    print("\n--- Single Person Simulation ---")
    
    person = Person(
        name="Zack",
        weight_lbs=165,
        sex='M',
        max_BAC=0.16,
        base_drinking_tendency=0.75
    )
    
    simulator = DrinkingSimulator(
        person=person,
        session_hours=6,
        decision_interval_minutes=10
    )
    
    # Run simulations
    print(f"\nSimulating {person.name}'s drinking behavior...")
    print(f"  Weight: {person.weight_lbs} lbs")
    print(f"  Sex: {person.sex}")
    print(f"  Max BAC: {person.max_BAC}")
    print(f"  Session length: {simulator.session_hours} hours")
    print(f"  Running 1000 simulations...\n")
    
    results = simulator.run_simulations(n_simulations=1000)
    stats = simulator.analyze_results(results)
    
    print("Results:")
    print(f"  Mean drinks: {stats['mean']:.2f}")
    print(f"  Median drinks: {stats['median']:.2f}")
    print(f"  Std deviation: {stats['std']:.2f}")
    print(f"  Range: {stats['min']:.0f} - {stats['max']:.0f}")
    print(f"  75th percentile: {stats['percentile_75']:.2f}")
    print(f"  90th percentile: {stats['percentile_90']:.2f}")
    
    # Plot results
    simulator.plot_results(results)