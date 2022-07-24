WITH ages as (
    SELECT a.subject_id, 
    ROUND( (CAST(EXTRACT(epoch FROM a.admittime - p.dob)/(60*60*24*365.242) AS numeric)), 4) AS age
        FROM mimiciii.admissions a
        INNER JOIN mimiciii.patients p
        ON a.subject_id = p.subject_id
)
SELECT *
    FROM ages
    ORDER BY subject_id ASC
