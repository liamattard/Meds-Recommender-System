SELECT hadm_id, ndc
    FROM mimiciii.prescriptions
    WHERE ndc is not null
