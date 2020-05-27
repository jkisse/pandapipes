import csv
import os

import pandas as pd
import requests
from pandapipes import pp_dir
import numpy as np

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


def download_as_csv_from_nist(chemical, tempr_k, p_min, p_max, p_step=1,
                              p_unit="bar", visc_unit="uPa*s", target_folder='pp_prop'):
    """
    Downloads chemical properties from the NIST Webhook.
     This query is for isothermal properties for various pressures.
     The web-API can be visited on https://webbook.nist.gov/chemistry/fluid/ .
    :param chemical: chemical name (long, not abbreviated)
    :type chemical: str
    :param tempr_k: temperature in Kelvin at which isothermal properties should be queried
    :type tempr_k: float, int, or str
    :param p_min: minimum pressure in bar for which the properties should be queried
    :type p_min: float, int
    :param p_max: maximum pressure in bar for which the properties should be queried
    :type p_max: float, int
    :param p_step: step size for pressures between p_min and p_max
    :type p_step: float, int
    :param p_unit: unit for pressures. Either "bar" or "MPa" (or 'atm.', 'torr', 'psia')
    :type p_unit: str
    :param p_unit: unit for viscosity. Either "Pa*s" or "uPa*a" (u = micro)
    :type p_unit: str
    :param target_folder: where the csv files should be saved.
                          default is pp_prop = pandapipes/properties/[chemical_name]
    :type target_folder: str
    :return: path to created csv file
    :rtype: str
    """
    # put new CAS numbers here, without dashes:
    cas_number_dict = {"hydrogen": "1333740",
                       "H2": "1333740",
                       "carbondioxide": "124389",
                       "CO2": "124389",
                       "methane": "74828"}

    if chemical not in cas_number_dict.keys():
        raise NotImplementedError("Chemical %s is not yet implemented in the chemical->ID dict."
                                  "Please note, that only full names are implemented.\n"
                                  "You can look up the CAS number for your chemical on "
                                  "Wikipedia or here https://www.commonchemistry.org/index.aspx "
                                  "and add it to the list (without dashes).")
    cas_number = cas_number_dict[chemical]
    if target_folder == "pp_prop":
        prop_folder = os.path.join(pp_dir, 'properties', chemical)
    else:
        prop_folder = target_folder
    if not os.path.exists(prop_folder):
        os.makedirs(prop_folder)
    target_csv = os.path.join(prop_folder, "NIST_" + str(tempr_k) + ".csv")

    url = r"https://webbook.nist.gov/cgi/fluid.cgi?Action=Data&Wide=on" \
          r"&ID=C" + str(cas_number) + \
          r"&Type=IsoTherm" \
          r"&Digits=8" \
          r"&PLow=" + str(p_min) + \
          r"&PHigh=" + str(p_max) + \
          r"&PInc=" + str(p_step) + \
          r"&T=" + str(tempr_k) + \
          r"&RefState=DEF" \
          r"&TUnit=K" \
          r"&PUnit=" + str(p_unit) + \
          r"&DUnit=kg%2Fm3" \
          r"&HUnit=kJ%2Fkg" \
          r"&WUnit=m%2Fs" \
          r"&VisUnit=" + str(visc_unit) + \
          r"&STUnit=N%2Fm"
    logger.info("Downloading data from %s." % url)
    myfile = requests.get(url)
    table_raw = myfile.content.decode("utf-8")
    table_raw = table_raw.replace('\t', ',')

    with open(target_csv, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for r in table_raw.split("\n"):
            writer.writerow(r.split(","))
    logger.debug("%s data for %.2f Kelvin stored in %s" % (chemical, tempr_k, prop_folder))
    return target_csv


def merge_csv_files_to_df(csv_files):
    """
    Merge csv files from a given list to a pandas dataframe.
    Rows with NA values will be completely dropped.
    :param csv_files: list of file paths to csv files
    :type csv_files: list
    :return: Merged Dataframe with infos from csv files
    :rtype: pandas.DataFrame
    """
    df = pd.DataFrame()
    for c in csv_files:
        df_new = pd.read_csv(c)
        df_new.dropna(axis="rows", inplace=True)
        df = df.append(df_new, ignore_index=True)
    logger.debug(str(len(csv_files)) + " csv files merged.")
    return df


def create_database_from_nist(chemical, t_min_k=273.15, t_max_k=423.15, t_step_k=10,
                              p_min=1, p_max=200, p_step=1, p_unit="bar", visc_unit="Pa*s",
                              target_folder='pp_prop', save_summary=False):
    """
    For the given chemical and p-T-boundaries, download data from NIST database and merge it.
        The web-API can be visited on https://webbook.nist.gov/chemistry/fluid/ .
    :param chemical: chemical name (long, not abbreviated)
    :type chemical: str
    :param t_min_k: minimum temperature in Kelvin for which the properties should be queried
    :type t_min_k: float, int
    :param t_max_k: maximum temperature in Kelvin for which the properties should be queried
    :type t_max_k: float, int
    :param t_step_k: step size for temperatures between t_min_k and t_max_k
    :type t_step_k: float, int
    :param p_min: minimum pressure in p_unit for which the properties should be queried
    :type p_min: float, int
    :param p_max: maximum pressure in p_unit for which the properties should be queried
    :type p_max: float, int
    :param p_step: step size for pressures between p_min and p_max
    :type p_step: float, int
    :param p_unit: unit for pressures. Either "bar" or "MPa" (or 'atm.', 'torr', 'psia')
    :type p_unit: str
    :param target_folder: where the csv files should be saved.
                          default is pp_prop = pandapipes/properties/[chemical_name]
    :type target_folder: str
    :param save_summary: Should the resulting Dataframe stored as csv?
    :type save_summary: boolean
    :return: Dataframe with pressure and temperature column and many chemical properties in the
             other columns
    :rtype: pandas.DataFrame
    """
    csv_files = []
    t = t_min_k
    while t <= t_max_k:  # avoiding range() to allow also floats for t_max_k
        csv_files = csv_files + [
            download_as_csv_from_nist(chemical, t, p_min, p_max, p_step, p_unit, visc_unit,
                                      target_folder)]
        t += t_step_k
    df = merge_csv_files_to_df(csv_files)
    if "Pressure (MPa)" in df.columns:
        df["Pressure (bar)"] = df["Pressure (MPa)"] * 10
    if save_summary:
        if target_folder == "pp_prop":
            df.to_csv(os.path.join(pp_dir, "properties", chemical, "NIST_summary.csv"), index=False)
        else:
            df.to_csv(os.path.join(target_folder, "NIST_summary.csv"), index=False)
        logger.debug("NIST_summary.csv saved.")
    return df


def merge_nist_csvs_in_folder(folder_path):
    """
    Merge all csv files in the folder_geojson if they start with "NIST".
    Convenience functions, if the csv files have already been downloaded and just need to be
    merged.
    :param folder_path: path to folder_geojson with csv files, e.g. in pandapipes/properties/[chem.name]
    :type folder_path: str
    :return: Merged Dataframe with infos from csv files
    :rtype: pandas.DataFrame
    """

    csvs = []
    for _, _, files in os.walk(folder_path):
        for f in files:
            if (f[:4] == "NIST") & (f[-4:] == ".csv"):
                csvs = csvs + [os.path.join(folder_path, f)]
    return merge_csv_files_to_df(csvs)


def create_textfiles_from_nist_summary(folder, at_pressure_bar=1):
    df = pd.read_csv(os.path.join(folder, "nist_summary.csv"))
    df = df.loc[df["Pressure (bar)"] == at_pressure_bar]
    pandapipes_names = {  # 'Temperature (K)': 'temperature', 'Pressure (MPa)',
        'Density (kg/m3)': 'density',
        'Cp (J/mol*K)': 'heat_capacity',
        'Viscosity (uPa*s)': 'viscosity'}
    df.rename(pandapipes_names, axis='columns', inplace=True)

    for prop in pandapipes_names.values():
        # isobar_1 = df.loc[df["Pressure (bar)"] == 1, ['Temperature (K)', prop]]
        isobar_1 = df[['Temperature (K)', prop]]
        arr = np.array(isobar_1)
        np.savetxt(os.path.join(folder, prop + ".txt"), arr)
        logger.debug(str(os.path.join(folder, prop + ".txt")) + " created.")


if __name__ == "__main__":
    chemical = "hydrogen"
    # chemical = "methane"
    create_database_from_nist(chemical, t_min_k=263.15, t_max_k=423.15, t_step_k=2,
                              p_min=1, p_max=240, p_step=1, save_summary=True)
