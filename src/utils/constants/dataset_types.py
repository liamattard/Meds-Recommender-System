import enum


class Dataset_Type(enum.Enum):

    full4Age = "Full Dataset with ATC4 codes and age"

    fullATC4 = "Full Dataset with ATC4 codes"
    full1VATC4 = "Full Dataset w patients that have only 1 visit and ATC4 codes"
    fullM1VATC4 = "Full Dataset w patients that have more than 1 visit and ATC4 codes"

    fullATC3 = "Full Dataset with ATC3 codes"
    full1VATC3 = "Full Dataset w patients that have only 1 visit and ATC3 codes"
    fullM1VATC3 = "Full Dataset w patients that have more than 1 visit and ATC3 codes"

    fullNDC = "Full Dataset with NDC codes"
    full1VNDC = "Full Dataset w patients that have only 1 visit and NDC codes"
    fullM1VNDC = "Full Dataset w patients that have more than 1 visit and NDC codes"

    sota = "S.O.T.A Dataset"

    experiment = "Experimental dataset type"
