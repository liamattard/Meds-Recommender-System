SELECT DISTINCT lower(drug)
    FROM mimiciii.prescriptions
    WHERE drug NOT LIKE ' '
