SELECT hadm_id, itemid
    FROM mimiciii.labevents
    WHERE drug NOT LIKE ' '
