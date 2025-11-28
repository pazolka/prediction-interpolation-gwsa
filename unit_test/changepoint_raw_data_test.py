import unittest
from pathlib import Path
import pandas as pd
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import utils.outliers_helpers as ohp
import utils.changepoint_helpers as chp

class ChangepointTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        csv_path = PROJECT_ROOT / "data" / "Bangladesh" / "target" / "raw_data.csv"
        ts_file = pd.read_csv(csv_path)
        ts_file['Date'] = pd.to_datetime(ts_file['Date'], format='%d/%m/%Y')
        # Correct years that are mistakenly in the 2000s
        ts_file['Date'] = ts_file['Date'].apply(lambda x: x.replace(year=x.year - 100) if x.year > 2025 else x)
        ts_file = ts_file.set_index('Date')
        cls.df = ts_file

    def test_shape(self):
        self.assertEqual(self.df.shape, (708, 1250))

    def test_detect_well_change_SY013(self):
        ts_raw = self.df['SY013']
        
        # --- Outlier Detection ---
        ts_cleaned, outlier_found = ohp.remove_outliers(ts_raw.copy())
        ts_cleaned2, filtered_breaks = chp.detect_well_change(ts_cleaned.copy())

        self.assertTrue(len(filtered_breaks) == 1)
        self.assertTrue(filtered_breaks[0] == 318)

    
    def test_detect_well_change_RJ034(self):
        ts_raw = self.df['RJ034']
        
        # --- Outlier Detection ---
        ts_cleaned, outlier_found = ohp.remove_outliers(ts_raw.copy())
        ts_cleaned2, filtered_breaks = chp.detect_well_change(ts_cleaned.copy())

        self.assertTrue(len(filtered_breaks) == 1)
        self.assertTrue(filtered_breaks[0] == 387)
        

    def test_detect_well_change_PA029(self):
        ts_raw = self.df['PA029']
        
        # --- Outlier Detection ---
        ts_cleaned, outlier_found = ohp.remove_outliers(ts_raw.copy())
        ts_cleaned2, filtered_breaks = chp.detect_well_change(ts_cleaned.copy())

        self.assertTrue(len(filtered_breaks) == 1)
        self.assertTrue(filtered_breaks[0] == 374)
        
    
    def test_detect_well_change_BO034(self):
        ts_raw = self.df['BO034']
        
        # --- Outlier Detection ---
        ts_cleaned, outlier_found = ohp.remove_outliers(ts_raw.copy())
        ts_cleaned2, filtered_breaks = chp.detect_well_change(ts_cleaned.copy())

        self.assertTrue(len(filtered_breaks) == 1)
        self.assertTrue(filtered_breaks[0] == 371)

    def test_detect_no_well_change_DH100(self):
        ts_raw = self.df['DH100']
        
        # --- Outlier Detection ---
        ts_cleaned, outlier_found = ohp.remove_outliers(ts_raw.copy())
        ts_cleaned2, filtered_breaks = chp.detect_well_change(ts_cleaned.copy())

        self.assertTrue(len(filtered_breaks) == 0)


    def test_detect_no_well_change_DI063(self):
        ts_raw = self.df['DI063']
        
        # --- Outlier Detection ---
        ts_cleaned, outlier_found = ohp.remove_outliers(ts_raw.copy())
        ts_cleaned2, filtered_breaks = chp.detect_well_change(ts_cleaned.copy())

        self.assertTrue(len(filtered_breaks) == 0)

    
    def test_detect_no_well_change_BO028(self):
        ts_raw = self.df['BO028']
        
        # --- Outlier Detection ---
        ts_cleaned, outlier_found = ohp.remove_outliers(ts_raw.copy())
        ts_cleaned2, filtered_breaks = chp.detect_well_change(ts_cleaned.copy())

        self.assertTrue(len(filtered_breaks) == 0)


    def test_detect_no_well_change_RJ091(self):
        ts_raw = self.df['RJ091']
         
        # --- Outlier Detection ---
        ts_cleaned, outlier_found = ohp.remove_outliers(ts_raw.copy())
        ts_cleaned2, filtered_breaks = chp.detect_well_change(ts_cleaned.copy())

        self.assertTrue(len(filtered_breaks) == 0)

    
    def test_detect_no_well_change_SY073(self):
        ts_raw = self.df['SY073']
        
        # --- Outlier Detection ---
        ts_cleaned, outlier_found = ohp.remove_outliers(ts_raw.copy())
        ts_cleaned2, filtered_breaks = chp.detect_well_change(ts_cleaned.copy())

        self.assertTrue(len(filtered_breaks) == 0)

    ### edge case: well dried out
    def test_detect_no_well_change_RJ136(self):
        ts_raw = self.df['RJ136']
        
        # --- Outlier Detection ---
        ts_cleaned, outlier_found = ohp.remove_outliers(ts_raw.copy())
        ts_cleaned2, filtered_breaks = chp.detect_well_change(ts_cleaned.copy())

        self.assertTrue(len(filtered_breaks) == 0)

    # def test_detect_well_change_SY047(self):
    #     ts_raw = self.df['SY047']
         
    #     # --- Outlier Detection ---
    #     ts_cleaned, outlier_found = ohp.remove_outliers(ts_raw.copy())
    #     ts_cleaned2, filtered_breaks = chp.detect_well_change(ts_cleaned.copy())

    #     self.assertTrue(len(filtered_breaks) > 0)

    def test_detect_well_change_DH124(self):
        ts_raw = self.df['DH124']
        
        # --- Outlier Detection ---
        ts_cleaned, outlier_found = ohp.remove_outliers(ts_raw.copy())
        ts_cleaned2, filtered_breaks = chp.detect_well_change(ts_cleaned.copy())

        self.assertTrue(len(filtered_breaks) == 1)
        self.assertTrue(filtered_breaks[0] == 613)

    def test_detect_well_change_DH002(self):
        ts_raw = self.df['DH002']
        
        # --- Outlier Detection ---
        ts_cleaned, outlier_found = ohp.remove_outliers(ts_raw.copy())
        ts_cleaned2, filtered_breaks = chp.detect_well_change(ts_cleaned.copy())

        self.assertTrue(len(filtered_breaks) == 1)
        self.assertTrue(filtered_breaks[0] == 679)

    # two sudden change points
    def test_detect_well_change_RJ126(self):
        ts_raw = self.df['RJ126']
         
        # --- Outlier Detection ---
        ts_cleaned, outlier_found = ohp.remove_outliers(ts_raw.copy())
        ts_cleaned2, filtered_breaks = chp.detect_well_change(ts_cleaned.copy())

        self.assertTrue(len(filtered_breaks) == 2)
        self.assertTrue(filtered_breaks[0] == 385)
        self.assertTrue(len(filtered_breaks) > 0)

    # change in the middle of the range
    def test_detect_well_change_SY026(self):
        ts_raw = self.df['SY026']
         
        # --- Outlier Detection ---
        ts_cleaned, outlier_found = ohp.remove_outliers(ts_raw.copy())
        ts, filtered_breaks = chp.detect_well_change(ts_cleaned.copy())

        self.assertTrue(len(filtered_breaks) == 2)
        self.assertTrue(filtered_breaks[0] == 282)
        self.assertTrue(filtered_breaks[1] == 557)

    # change at the end of the period
    def test_detect_well_change_CT042(self):
        ts_raw = self.df['CT042']
         
        # --- Outlier Detection ---
        ts_cleaned, outlier_found = ohp.remove_outliers(ts_raw.copy())
        ts_cleaned2, filtered_breaks = chp.detect_well_change(ts_cleaned.copy())

        self.assertTrue(len(filtered_breaks) == 1)
        self.assertTrue(filtered_breaks[0] == 520)


if __name__ == "__main__":
    unittest.main()
