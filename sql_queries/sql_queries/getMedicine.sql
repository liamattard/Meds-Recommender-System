SELECT hadm_id, ndc, lower(drug)
    FROM mimiciii.prescriptions
    WHERE ndc is not null
