SELECT DISTINCT drug
    FROM mimiciii.prescriptions
    WHERE drug NOT LIKE ' '
