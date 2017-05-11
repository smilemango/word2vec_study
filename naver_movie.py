import urllib
import urllib.request
import urllib.parse
import bs4
import re
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor


def deleteTag(x):
    return re.sub("<[^>]*>", "", x)


def getComments(code):
    def makeArgs(code, page):
        params = {
            'code': code,
            'type': 'after',
            'isActualPointWriteExecute': 'false',
            'isMileageSubscriptionAlready': 'false',
            'isMileageSubscriptionReject': 'false',
            'page': page
        }
        return urllib.parse.urlencode(params)

    def innerHTML(s, sl=0):
        ret = ''
        for i in s.contents[sl:]:
            if i is str:
                ret += i.strip()
            else:
                ret += str(i)
        return ret

    def fText(s):
        if len(s): return innerHTML(s[0]).strip()
        return ''

    retList = []
    colSet = set()
    print("Processing: %d" % code)
    page = 1
    while 1:
        try:
            f = urllib.request.urlopen(
                "http://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?" + makeArgs(code, page))
            data = f.read().decode('utf-8')
        except:
            break
        soup = bs4.BeautifulSoup(re.sub("&#(?![0-9])", "", data), "html.parser")
        cs = soup.select(".score_result li")
        if not len(cs): break
        for link in cs:
            try:
                url = link.select('.score_reple em a')[0].get('onclick')
            except:
                print(page)
                print(data)
                raise ""
            m = re.search('[0-9]+', url)
            if m:
                url = m.group(0)
            else:
                url = ''
            if url in colSet: return retList
            colSet.add(url)
            cat = fText(link.select('.star_score em'))
            cont = fText(link.select('.score_reple p'))
            cont = re.sub('<span [^>]+>.+?</span>', '', cont)
            retList.append((url, cat, cont))
        page += 1

    return retList


def getTitleYear(code):

    print("Processing: %d" % code)
    try:
        f = urllib.request.urlopen(
            "http://movie.naver.com/movie/bi/mi/basic.nhn?code=" + str(code) )
        data = f.read().decode('utf-8')

        soup = bs4.BeautifulSoup(re.sub("&#(?![0-9])", "", data), "html.parser")
        title_kr = soup.select('.mv_info_area')[0].find_all('h3', class_='h_movie')[0].text.strip()
        movie_year = soup.select('.mv_info_area')[0].find_all('strong', class_='h_movie2')[0].text.split(',')[-1].strip()
        title_en = ','.join(soup.select('.mv_info_area')[0].find_all('strong', class_='h_movie2')[0].text.split(',')[0:-1]).strip()
    except :
        print(traceback.format_stack())


    return title_kr, title_en, movie_year


def fetch_movie_info(i):
    outname = 'info/%d.txt' % i
    try:
        if os.stat(outname).st_size > 0 :
            print("file alread exist.")
            return #이미 존재하면 아무것도 안한다
    except:
        None

    title_kr, title_en, year = getTitleYear(i)

    f = open(outname, 'w', encoding='utf-8')
    f.write("INSERT INTO movie VALUES ('%s','%s','%s'); " % (title_kr, title_en, year))
    f.write('\n')
    f.close()
    time.sleep(1)


def fetch(i):
    outname = 'comments/%d.txt' % i
    try:
        if os.stat(outname).st_size > 0: return
    except:
        None
    rs = getComments(i)
    if not len(rs): return
    f = open(outname, 'w', encoding='utf-8')
    f.write('INSERT INTO comments VALUES ')
    for idx, r in enumerate(rs):
        if idx: f.write(',\n')
        f.write("(%d,%s,%s,'%s')" % (i, r[0], r[1], r[2].replace("'", "''").replace("\\", "\\\\")))
    f.write(';\n')
    f.close()
    time.sleep(1)


with ThreadPoolExecutor(max_workers=5) as executor:
    for i in range(121048, 121049):
        executor.submit(fetch, i)


# with ThreadPoolExecutor(max_workers=1) as executor:
#     for i in range(121048, 121049):
#         executor.submit(fetch_movie_info,i)

