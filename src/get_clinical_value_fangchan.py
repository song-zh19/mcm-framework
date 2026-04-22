import json


inames = json.load(open("/home/user/code/stat/logs/inames.json"))
workno2zhibiao = json.load(open("/home/user/code/stat/logs/workno2zhibiao_new.json"))
used_worknos = json.load(open("/home/user/code/stat/logs/used_worknos.json"))
# workno2filename = json.load(open("/home/user/code/stat/logs/workno2filename.json"))

clinical_value = {}
for workno in workno2zhibiao:
    if workno in used_worknos:
        zhibiao = workno2zhibiao[workno]
        zhibiao_iname = zhibiao['iname']
        zhibiao_iresultvalue = zhibiao['iresultvalue']
        assert len(zhibiao_iname) == len(zhibiao_iresultvalue)
        values = [None for _ in range(len(inames))]
        for idx in range(len(zhibiao_iresultvalue)):
            val = None
            try:
                val = float(zhibiao_iresultvalue[idx])
            except:
                if zhibiao_iresultvalue[idx] == '阴性':
                    val = 0
                if zhibiao_iresultvalue[idx] == '阳性':
                    val = 1
            values[inames.index(zhibiao_iname[idx])] = val
        clinical_value[workno] = values

with open("/home/user/code/stat/logs/clinical_value.json", "w") as f:
    f.write(json.dumps(clinical_value))
