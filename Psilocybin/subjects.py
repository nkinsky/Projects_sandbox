import getpass
from pathlib import Path
import platform
from os import environ

if "kinsky" in getpass.getuser():
    if platform.system() == "Darwin":
        base_dir = Path("/Users/nkinsky/Documents/UM/Working/Psilocybin/Recording_Rats")
    elif environ["HOSTNAME"] == "lnx00004":
        base_dir = Path("/data3/Psilocybin/Recording_Rats")
else:
    assert False, "Base directory not yet set for users other than nkinsky"

chan_dict = {
    "Rey":   {"Saline1": 21, "Psilocybin": 21, "Saline2": 22},
    "Finn":  {"Saline1": 27, "Psilocybin": 27, "Saline2": 27},
    "Rose":  {"Saline1": 27, "Psilocybin": 26, "Saline2": 25},
    "Finn2": {"Saline1": 4,  "Psilocybin": 4,  "Saline2": 4}
}


class Finn:
    def __init__(self, base_dir=base_dir):
        self.base_dir = Path(base_dir)
        self.animal = "Finn"
        self.animal_num = 1
        self.sess_dict = {"Saline1": "2022_02_15_saline1",
                          "Psilocybin": "2022_02_17_psilocybin",
                          "Saline2": "2022_02_18_saline2"}


class Rey:
    def __init__(self, base_dir=base_dir):
        self.base_dir = Path(base_dir)
        self.animal = "Rey"
        self.animal_num = 2
        self.sess_dict = {"Saline1": "2022_06_01_saline1",
                          "Psilocybin": "2022_06_02_psilocybin",
                          "Saline2": "2022_06_03_saline2"}


class Rose:
    def __init__(self, base_dir=base_dir):
        self.base_dir = Path(base_dir)
        self.animal = "Rose"
        self.animal_num = 3
        self.sess_dict = {"Saline1": "2022_08_09_saline1",
                          "Psilocybin": "2022_08_10_psilocybin",
                          "Saline2": "2022_08_11_saline2"}


class Finn2:
    def __init__(self, base_dir=base_dir):
        self.base_dir = Path(base_dir)
        self.animal = "Finn2"
        self.animal_num = 4
        self.sess_dict = {"Saline1": "2023_05_24_saline1",
                          "Psilocybin": "2023_05_25_psilocybin",
                          "Saline2": "2023_05_26_saline2"}


class RecDir:
    def __init__(self, base_dir=base_dir):
        self.base_dir = Path(base_dir)

    @property
    def finn(self):
        return Finn(self.base_dir)

    @property
    def rey(self):
        return Rey(self.base_dir)

    @property
    def rose(self):
        return Rose(self.base_dir)

    @property
    def finn2(self):
        return Finn2(self.base_dir)


def get_psi_dir(animal_name, session_name):
    """Get working directory for animal_name + session_name name combo.

    Adjusts for upper-lower case and underscores in saline_1 and saline_2"""

    animal = getattr(RecDir(), animal_name.lower())  # load in animal data session

    session_name = "".join(session_name.split("_"))  # get rid of underscore

    return animal.base_dir / animal.animal / animal.sess_dict[session_name.capitalize()]

def get_pyr_ch(animal_name, session_name: str in ["Saline1", "Psilocybin", "Saline2"]):
    """Gets pyramidal cell channel to use."""
    assert session_name in ["Saline1", "Psilocybin", "Saline2"]
    return chan_dict[animal_name][session_name]

