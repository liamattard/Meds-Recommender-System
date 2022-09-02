SELECT DISTINCT lower(drug), ndc
    FROM mimiciii.prescriptions
    ORDER BY lower
