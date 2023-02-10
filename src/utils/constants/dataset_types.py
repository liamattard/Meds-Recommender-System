import enum


class Dataset_Type(enum.Enum):

    # Objective 1
    realistic4 = "Realistic Dataset with visit split ATC4"
    realistic3 = "Realistic Dataset with visit split ATC3"
    realisticNDC = "Realistic Dataset with visit split NDC"

    # Objective 3
    realisticNoPro3 = "Realistic Dataset with visit split ATC3 and rows with empty procedures"

    multiRealisticNoPro3 = "Realistic Dataset with visit split ATC3 and rows with empty procedures and only patients more than 1"

    full4Age = "Full Dataset with ATC4 codes and age"
    full3Age = "Full Dataset with ATC3 codes and age"

    fullATC4= "Full Dataset with ATC4 codes"
    full1VATC4 = "Full Dataset w patients that have only 1 visit and ATC4 codes"
    fullM1VATC4 = "Full Dataset w patients that have more than 1 visit and ATC4 codes"

    fullATC3 = "Full Dataset with ATC3 codes"
    full1VATC3 = "Full Dataset w patients that have only 1 visit and ATC3 codes"
    fullM1VATC3 = "Full Dataset w patients that have more than 1 visit and ATC3 codes"

    fullNDC = "Full Dataset with NDC codes"
    full1VNDC = "Full Dataset w patients that have only 1 visit and NDC codes"
    fullM1VNDC = "Full Dataset w patients that have more than 1 visit and NDC codes"

    sota = "State of the art dataset used by GameNet/SafeDrug with the fixed ehr_ad"
    old_sota = "State of the art dataset used by GameNet/SafeDrug"
    sota_single_only = "State of the art dataset but with only single visits"
    sota_with_single = "State of the art dataset inlcuding single + sota"

    all_3 = "realistic 3 and all of the newly added items"
    all_4 = "realistic 4 and all of the newly added items"

    experiment = "Experimental dataset type"
