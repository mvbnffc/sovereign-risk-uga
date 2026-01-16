import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
from tqdm import tqdm
import time
import os
import shutil
import warnings


class ResilientDIGNADPrecomputer:
    """
    Robust DIGNAD pre-computation with automatic checkpointing and zero-value handling
    """
    
    def __init__(self, param_grid, output_path, checkpoint_dir=None, 
                 save_interval=10, min_threshold=0.01):
        """
        Initialize resilient pre-computer
        
        Parameters:
        -----------
        param_grid : pd.DataFrame
            Parameter combinations to simulate
        output_path : str
            Final output file path
        checkpoint_dir : str
            Directory for checkpoint files (default: creates one next to output)
        save_interval : int
            Save checkpoint every N simulations
        min_threshold : float
            Minimum value threshold - values below this are skipped or set to this
        """
        self.param_grid = param_grid
        self.output_path = Path(output_path)
        self.save_interval = save_interval
        self.min_threshold = min_threshold
        
        # Set up checkpoint directory
        if checkpoint_dir is None:
            self.checkpoint_dir = self.output_path.parent / f"{self.output_path.stem}_checkpoints"
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint files
        self.progress_file = self.checkpoint_dir / "progress.json"
        self.results_file = self.checkpoint_dir / "results_checkpoint.pkl"
        self.backup_results_file = self.checkpoint_dir / "results_checkpoint_backup.pkl"
        self.skipped_file = self.checkpoint_dir / "skipped_simulations.json"
        
        # Track different types of outcomes
        self.completed_indices = set()
        self.failed_indices = set()
        self.skipped_indices = set()  # For zero/near-zero values
        self.results = []
        
        self._load_checkpoint()
    
    def _check_parameters_valid(self, row):
        """
        Check if parameters are valid for DIGNAD simulation
        
        Parameters:
        -----------
        row : pd.Series
            Parameter row to check
            
        Returns:
        --------
        is_valid : bool
            Whether parameters are valid
        reason : str
            Reason if invalid
        """
        # Check for zero or near-zero values in critical parameters
        critical_params = ['tradable_impact', 'nontradable_impact', 
                          'private_impact', 'public_impact']
        
        for param in critical_params:
            if param in row:
                value = row[param]
                
                # Check if value is effectively zero
                if pd.isna(value):
                    return False, f"{param} is NaN"
                
                if abs(value) < self.min_threshold:
                    return False, f"{param} = {value:.6f} is below threshold {self.min_threshold}"
        
        # Check if ALL parameters are zero (complete no-impact scenario)
        if all(abs(row.get(param, 1)) < self.min_threshold for param in critical_params):
            return False, "All impact parameters are zero"
        
        return True, "Valid"
    
    def _adjust_parameters(self, row, adjustment_method='threshold'):
        """
        Adjust parameters to avoid DIGNAD failures
        
        Parameters:
        -----------
        row : pd.Series
            Parameter row to adjust
        adjustment_method : str
            'threshold': Set minimum values to threshold
            'scale': Scale up all values proportionally
            'skip': Don't adjust, will skip this simulation
            
        Returns:
        --------
        adjusted_row : pd.Series
            Adjusted parameters
        was_adjusted : bool
            Whether adjustments were made
        """
        if adjustment_method == 'skip':
            return row, False
        
        adjusted_row = row.copy()
        was_adjusted = False
        
        critical_params = ['tradable_impact', 'nontradable_impact', 
                          'private_impact', 'public_impact']
        
        if adjustment_method == 'threshold':
            # Set any near-zero values to the minimum threshold
            for param in critical_params:
                if param in adjusted_row:
                    if abs(adjusted_row[param]) < self.min_threshold:
                        # Preserve sign but ensure minimum magnitude
                        sign = np.sign(adjusted_row[param]) if adjusted_row[param] != 0 else 1
                        adjusted_row[param] = sign * self.min_threshold
                        was_adjusted = True
        
        elif adjustment_method == 'scale':
            # Scale all values up if any are below threshold
            min_value = min(abs(adjusted_row[param]) for param in critical_params if param in adjusted_row)
            if min_value < self.min_threshold and min_value > 0:
                scale_factor = self.min_threshold / min_value
                for param in critical_params:
                    if param in adjusted_row:
                        adjusted_row[param] *= scale_factor
                was_adjusted = True
        
        return adjusted_row, was_adjusted
    
    def _load_checkpoint(self):
        """Load existing checkpoint if available"""
        if self.progress_file.exists():
            print(f"Found existing checkpoint at {self.checkpoint_dir}")
            try:
                with open(self.progress_file, 'r') as f:
                    checkpoint = json.load(f)
                
                self.completed_indices = set(checkpoint.get('completed_indices', []))
                self.failed_indices = set(checkpoint.get('failed_indices', []))
                self.skipped_indices = set(checkpoint.get('skipped_indices', []))
                
                # Load results
                if self.results_file.exists():
                    with open(self.results_file, 'rb') as f:
                        self.results = pickle.load(f)
                
                print(f"Resuming from checkpoint:")
                print(f"  - Completed: {len(self.completed_indices)}/{len(self.param_grid)} simulations")
                print(f"  - Failed: {len(self.failed_indices)} simulations")
                print(f"  - Skipped (zero values): {len(self.skipped_indices)} simulations")
                print(f"  - Last update: {checkpoint.get('last_update', 'Unknown')}")
                
                return True
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                print("Starting fresh...")
                self.completed_indices = set()
                self.failed_indices = set()
                self.skipped_indices = set()
                self.results = []
        
        return False
    
    def _save_checkpoint(self, force=False):
        """Save current progress to checkpoint files"""
        try:
            # Save progress JSON
            checkpoint = {
                'completed_indices': list(self.completed_indices),
                'failed_indices': list(self.failed_indices),
                'skipped_indices': list(self.skipped_indices),
                'total_simulations': len(self.param_grid),
                'last_update': datetime.now().isoformat(),
                'output_path': str(self.output_path),
                'save_interval': self.save_interval,
                'min_threshold': self.min_threshold
            }
            
            with open(self.progress_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            # Save skipped simulations details
            if self.skipped_indices:
                skipped_details = []
                for idx in self.skipped_indices:
                    row = self.param_grid.iloc[idx]
                    is_valid, reason = self._check_parameters_valid(row)
                    skipped_details.append({
                        'index': int(idx),
                        'reason': reason,
                        'parameters': row.to_dict()
                    })
                
                with open(self.skipped_file, 'w') as f:
                    json.dump(skipped_details, f, indent=2)
            
            # Backup existing results file before overwriting
            if self.results_file.exists():
                shutil.copy2(self.results_file, self.backup_results_file)
            
            # Save results
            with open(self.results_file, 'wb') as f:
                pickle.dump(self.results, f)
            
            if force:
                total_processed = len(self.completed_indices) + len(self.skipped_indices) + len(self.failed_indices)
                print(f"\nCheckpoint saved: {total_processed}/{len(self.param_grid)} processed")
                print(f"  - Completed: {len(self.completed_indices)}")
                print(f"  - Skipped: {len(self.skipped_indices)}")
                print(f"  - Failed: {len(self.failed_indices)}")
        
        except Exception as e:
            print(f"\nWarning: Could not save checkpoint: {e}")
    
    def run_simulation(self, idx, row, run_DIGNAD_func, sim_params):
        """
        Run a single DIGNAD simulation with error handling
        
        Parameters:
        -----------
        idx : int
            Index in parameter grid
        row : pd.Series
            Parameters for this simulation
        run_DIGNAD_func : callable
            DIGNAD simulation function
        sim_params : dict
            Additional simulation parameters
        
        Returns:
        --------
        dict : Simulation results or None if failed
        """
        try:
            # Run DIGNAD with parameters from grid
            gdp_impact, years = run_DIGNAD_func(
                sim_params['sim_start_year'],
                sim_params['nat_disaster_year'],
                sim_params['recovery_period'],
                row['tradable_impact'],
                row['nontradable_impact'],
                row.get('reconstruction_efficiency', 0),
                row.get('public_debt_premium', 0),
                row['private_impact'],
                row['public_impact'],
                row.get('share_tradable', 0.5)
            )
            
            # Store results
            result = row.to_dict()
            result['simulation_idx'] = idx
            result['gdp_trajectory'] = gdp_impact.tolist()
            result['years'] = years.tolist()
            result['max_gdp_loss'] = float(np.min(gdp_impact))
            result['cumulative_loss'] = float(np.sum(gdp_impact[gdp_impact < 0]))
            result['recovery_year'] = int(years[np.argmax(gdp_impact >= -0.5)]) if any(gdp_impact >= -0.5) else int(years[-1])
            result['was_adjusted'] = row.get('was_adjusted', False)
            
            return result
            
        except Exception as e:
            # Detailed error logging
            error_msg = f"Error in simulation {idx}: {str(e)}"
            if "division" in str(e).lower() or "zero" in str(e).lower():
                error_msg += " (likely due to zero/near-zero parameters)"
            print(f"\n{error_msg}")
            return None
    
    def precompute(self, run_DIGNAD_func, sim_params, 
                  handle_zeros='skip', max_retries=1):
        """
        Run pre-computation with automatic checkpointing and zero handling
        
        Parameters:
        -----------
        run_DIGNAD_func : callable
            DIGNAD simulation function
        sim_params : dict
            Dictionary with: sim_start_year, nat_disaster_year, recovery_period
        handle_zeros : str
            How to handle zero/near-zero values:
            - 'skip': Skip these simulations (mark as NaN)
            - 'threshold': Set to minimum threshold value
            - 'scale': Scale all parameters proportionally
            - 'try_anyway': Attempt simulation anyway
        max_retries : int
            Maximum retries for failed simulations
        
        Returns:
        --------
        pd.DataFrame : Complete results
        """
        # Determine which simulations to run
        all_processed = self.completed_indices | self.failed_indices | self.skipped_indices
        indices_to_run = [i for i in range(len(self.param_grid)) if i not in all_processed]
        
        if not indices_to_run:
            print("All simulations already processed!")
            return self._finalize_results()
        
        print(f"\nStarting DIGNAD pre-computation:")
        print(f"  - Total simulations: {len(self.param_grid)}")
        print(f"  - Already completed: {len(self.completed_indices)}")
        print(f"  - Already skipped (zeros): {len(self.skipped_indices)}")
        print(f"  - Already failed: {len(self.failed_indices)}")
        print(f"  - To process: {len(indices_to_run)}")
        print(f"  - Zero handling: {handle_zeros}")
        print(f"  - Min threshold: {self.min_threshold}")
        print(f"  - Checkpoint interval: every {self.save_interval} simulations")
        
        estimated_hours = len(indices_to_run) / 60
        print(f"  - Estimated time: {estimated_hours:.1f} hours")
        
        # Progress bar
        pbar = tqdm(total=len(self.param_grid), 
                   initial=len(all_processed),
                   desc="Processing simulations")
        
        try:
            for count, idx in enumerate(indices_to_run):
                row = self.param_grid.iloc[idx].copy()
                
                # Check if parameters are valid
                is_valid, reason = self._check_parameters_valid(row)
                
                if not is_valid and handle_zeros == 'skip':
                    # Skip this simulation
                    self.skipped_indices.add(idx)
                    
                    # Add a placeholder result with NaN values
                    result = row.to_dict()
                    result['simulation_idx'] = idx
                    result['max_gdp_loss'] = np.nan
                    result['cumulative_loss'] = np.nan
                    result['recovery_year'] = np.nan
                    result['gdp_trajectory'] = []
                    result['years'] = []
                    result['skip_reason'] = reason
                    result['was_skipped'] = True
                    
                    self.results.append(result)
                    pbar.update(1)
                    continue
                
                elif not is_valid and handle_zeros in ['threshold', 'scale']:
                    # Adjust parameters
                    row, was_adjusted = self._adjust_parameters(row, handle_zeros)
                    row['was_adjusted'] = was_adjusted
                    if was_adjusted:
                        row['adjustment_method'] = handle_zeros
                        row['original_valid'] = False
                
                # Run simulation with retries
                result = None
                for retry in range(max_retries):
                    if retry > 0:
                        print(f"\nRetrying simulation {idx} (attempt {retry + 1}/{max_retries})")
                    
                    result = self.run_simulation(idx, row, run_DIGNAD_func, sim_params)
                    
                    if result is not None:
                        break
                    
                    # If failed and we haven't tried adjusting yet
                    if retry == 0 and handle_zeros == 'threshold' and not row.get('was_adjusted', False):
                        print(f"  Trying with threshold adjustment...")
                        row, was_adjusted = self._adjust_parameters(row, 'threshold')
                        row['was_adjusted'] = True
                        row['adjustment_method'] = 'threshold_after_fail'
                    
                    time.sleep(0.5)  # Brief pause before retry
                
                if result is not None:
                    self.results.append(result)
                    self.completed_indices.add(idx)
                else:
                    # Failed even after retries
                    self.failed_indices.add(idx)
                    
                    # Add placeholder result
                    result = row.to_dict()
                    result['simulation_idx'] = idx
                    result['max_gdp_loss'] = np.nan
                    result['cumulative_loss'] = np.nan
                    result['recovery_year'] = np.nan
                    result['gdp_trajectory'] = []
                    result['years'] = []
                    result['was_failed'] = True
                    result['fail_reason'] = 'DIGNAD simulation failed'
                    
                    self.results.append(result)
                
                pbar.update(1)
                
                # Save checkpoint at intervals
                if (count + 1) % self.save_interval == 0:
                    self._save_checkpoint()
                    
                    # Also save intermediate full results every 100 simulations
                    if (count + 1) % 100 == 0:
                        self._save_intermediate_results()
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user! Saving checkpoint...")
            self._save_checkpoint(force=True)
            print("Checkpoint saved. Run again to resume.")
            raise
        
        except Exception as e:
            print(f"\n\nError occurred: {e}")
            print("Saving checkpoint before exit...")
            self._save_checkpoint(force=True)
            raise
        
        finally:
            pbar.close()
        
        # Final save
        self._save_checkpoint(force=True)
        
        return self._finalize_results()
    
    def _save_intermediate_results(self):
        """Save intermediate results to a separate file"""
        if self.results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            n_results = len(self.results)
            
            intermediate_path = self.checkpoint_dir / f"results_intermediate_{n_results}_{timestamp}.pkl"
            df = pd.DataFrame(self.results)
            df.to_pickle(intermediate_path)
            
            # Also save as CSV for inspection (without trajectories)
            csv_path = self.checkpoint_dir / f"results_intermediate_{n_results}_{timestamp}.csv"
            df_csv = df.drop(columns=['gdp_trajectory', 'years'], errors='ignore')
            df_csv.to_csv(csv_path, index=False)
            
            print(f"\n  Intermediate results saved ({n_results} simulations)")
    
    def _finalize_results(self):
        """Finalize and save complete results"""
        print(f"\nFinalizing results...")
        
        # Create DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Sort by original index
        if 'simulation_idx' in results_df.columns:
            results_df = results_df.sort_values('simulation_idx').reset_index(drop=True)
        
        # Calculate statistics
        n_completed = len(self.completed_indices)
        n_skipped = len(self.skipped_indices)
        n_failed = len(self.failed_indices)
        n_total = len(self.param_grid)
        
        print(f"\nSimulation Summary:")
        print(f"  Total simulations: {n_total}")
        print(f"  Successfully completed: {n_completed} ({n_completed/n_total*100:.1f}%)")
        print(f"  Skipped (zero values): {n_skipped} ({n_skipped/n_total*100:.1f}%)")
        print(f"  Failed: {n_failed} ({n_failed/n_total*100:.1f}%)")
        
        # Save final results
        if self.output_path.suffix == '.pkl':
            results_df.to_pickle(self.output_path)
        elif self.output_path.suffix == '.csv':
            # For CSV, save without trajectory columns
            df_csv = results_df.drop(columns=['gdp_trajectory', 'years'], errors='ignore')
            df_csv.to_csv(self.output_path, index=False)
            
            # Also save full results as pickle
            pkl_path = self.output_path.with_suffix('.pkl')
            results_df.to_pickle(pkl_path)
            print(f"Full results (with trajectories) saved to: {pkl_path}")
        
        print(f"Results saved to: {self.output_path}")
        
        # Save detailed report
        report_path = self.checkpoint_dir / "simulation_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"DIGNAD Pre-computation Report\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Output file: {self.output_path}\n\n")
            f.write(f"Summary:\n")
            f.write(f"  Total simulations: {n_total}\n")
            f.write(f"  Completed: {n_completed}\n")
            f.write(f"  Skipped: {n_skipped}\n")
            f.write(f"  Failed: {n_failed}\n\n")
            
            if n_completed > 0:
                valid_results = results_df[~results_df['max_gdp_loss'].isna()]
                if len(valid_results) > 0:
                    f.write(f"Results Statistics:\n")
                    f.write(f"  Max GDP Loss:\n")
                    f.write(f"    Mean: {valid_results['max_gdp_loss'].mean():.3f}%\n")
                    f.write(f"    Std: {valid_results['max_gdp_loss'].std():.3f}%\n")
                    f.write(f"    Min: {valid_results['max_gdp_loss'].min():.3f}%\n")
                    f.write(f"    Max: {valid_results['max_gdp_loss'].max():.3f}%\n")
        
        print(f"Report saved to: {report_path}")
        
        return results_df


def precompute_dignad_surface(param_grid, save_path, 
                              sim_start_year=2022, nat_disaster_year=2027, 
                              recovery_period=3, run_DIGNAD=None,
                              save_interval=10, checkpoint_dir=None,
                              handle_zeros='skip', min_threshold=0.01):
    """
    Convenient wrapper function for resilient DIGNAD pre-computation with zero handling
    
    Parameters:
    -----------
    param_grid : pd.DataFrame
        Parameter combinations to simulate
    save_path : str
        Output file path (.csv or .pkl)
    sim_start_year : int
        Simulation start year
    nat_disaster_year : int
        Year of disaster event
    recovery_period : int
        Recovery period in years
    run_DIGNAD : callable
        DIGNAD simulation function
    save_interval : int
        Save checkpoint every N simulations
    checkpoint_dir : str
        Custom checkpoint directory (optional)
    handle_zeros : str
        How to handle zero/near-zero values:
        - 'skip': Skip these simulations (save as NaN)
        - 'threshold': Set to minimum threshold value
        - 'scale': Scale all parameters proportionally
        - 'try_anyway': Attempt simulation anyway
    min_threshold : float
        Minimum value threshold for parameters
    
    Returns:
    --------
    pd.DataFrame : Simulation results
    """
    
    # Check if run_DIGNAD is provided
    if run_DIGNAD is None:
        from sovereign.macroeconomic import run_DIGNAD
    
    # Create simulation parameters
    sim_params = {
        'sim_start_year': sim_start_year,
        'nat_disaster_year': nat_disaster_year,
        'recovery_period': recovery_period
    }
    
    # Initialize resilient pre-computer
    precomputer = ResilientDIGNADPrecomputer(
        param_grid=param_grid,
        output_path=save_path,
        checkpoint_dir=checkpoint_dir,
        save_interval=save_interval,
        min_threshold=min_threshold
    )
    
    # Run pre-computation
    results = precomputer.precompute(
        run_DIGNAD, 
        sim_params,
        handle_zeros=handle_zeros,
        max_retries=1
    )
    
    # Print summary of handling
    n_skipped = results['was_skipped'].sum() if 'was_skipped' in results.columns else 0
    n_adjusted = results['was_adjusted'].sum() if 'was_adjusted' in results.columns else 0
    n_failed = results['was_failed'].sum() if 'was_failed' in results.columns else 0
    
    print("\n" + "=" * 50)
    print("ZERO/FAILURE HANDLING SUMMARY")
    print("=" * 50)
    print(f"Skipped (zero values): {n_skipped}")
    print(f"Adjusted parameters: {n_adjusted}")
    print(f"Failed simulations: {n_failed}")
    print(f"Successful simulations: {len(results) - n_skipped - n_failed}")
    
    return results


if __name__ == "__main__":
    print("Resilient DIGNAD Pre-computation with Zero Handling")
    print("=" * 60)
    print("\nFeatures:")
    print("  ✓ Automatic checkpointing")
    print("  ✓ Resume from interruption")
    print("  ✓ Handles zero/near-zero parameters")
    print("  ✓ Saves NaN for failed simulations")
    print("  ✓ Detailed progress tracking")
    print("\nZero handling options:")
    print("  - 'skip': Skip and save as NaN")
    print("  - 'threshold': Set to minimum threshold")
    print("  - 'scale': Scale parameters proportionally")
    print("  - 'try_anyway': Attempt anyway (may fail)")
