SELECT "hadm_id", "valuenum" FROM mimiciii.chartevents
where "itemid" = 211
and "valuenum" is not NULL
and "valuenum" != 0
ORDER BY "hadm_id"
