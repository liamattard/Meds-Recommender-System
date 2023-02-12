select subject_id,
    regexp_replace(split_part(split_part(split_part(diagnosis, '/',1), ',',1), ';', 1), '[^a-zA-Z ]', '', 'g') as diagnose
from mimiciii.admissions
order by subject_id