import re

fr = open("./middleman.log")
#fr = open("./sample1.txt")
fw = open("./data.csv","w")

# O[0-9]{7}[ ]*([^ ]*)[ ]*O[0-9]{7} -> 사람 이름 뽑기
# ([0-9]{2}\-[A-Z]{3}\-[0-9]{2}) -> 날짜
# ([0-9]{2}\-[A-Z]{3}\-[0-9]{2})[ ]*(.*) -> 그룹2 : 쿼리


line_to_write_front = ""
fw.write("사번,작업일,쿼리(일부)\n")
while True:

    line = fr.readline()
    # O1205727        조현준                                   O1205727                       toad.exe                       SEL 31-MAR-17    SELECT BASE_DATE              BASE_DATE               /*    기준_일자
    if not line:
        break

    # 좌우 공백 지우기
    line = line.strip()
    if line == "":
        continue

    emp_no = re.findall('(^O[0-9]{7}|^T[0-9]{7}|^K[0-9]{6})', line)

    if line_to_write_front != "":
        if len(emp_no) == 0:
            fw.write("\n"+ line.replace('"','""')  )
            continue
        else:
            fw.write( '"\n')

    date = re.findall('([0-9]{2}\-[A-Z]{3}\-[0-9]{2})', line)
    query = re.findall('([0-9]{2}\-[A-Z]{3}\-[0-9]{2})[ ]*(.*)', line)

    if len(emp_no) > 0:
        line_to_write_front = emp_no[0] + "," + date[0] + "," + '"' + query[0][1].replace('"','""')
        fw.write(line_to_write_front)
        emp_no = None
        date = None
        query = None
fr.close()
fw.close()
