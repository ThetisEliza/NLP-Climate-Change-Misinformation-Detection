import os
import random
import re
import sys
import time
import urllib
import json
from collections import deque
from urllib.parse import urlsplit, urljoin

import requests
import configparser
from bs4 import BeautifulSoup
from nltk import sent_tokenize
from nltk import word_tokenize
import lxml
from tqdm import tqdm


import os
import random
import re
import sys
import time
import urllib
import json
from collections import deque
from urllib.parse import urlsplit, urljoin

import requests
import configparser
from bs4 import BeautifulSoup
from nltk import sent_tokenize
from nltk import word_tokenize
import lxml
from tqdm import tqdm


class ArticleCrawler:
    def __init__(self, silent=True):
        self.user_agent = self._read_user_agent()
        self.silent = silent
        self._read_previous()

    def _read_previous(self):

        try:
            print('Reading from previous')
            with open('neg.json', 'r') as f:
                self.d = json.load(f)
            self.article_set = {v['text'] for v in self.d.values()}
            self.num = len(self.d)
            print('Reading succeed')
        except:
            print('Reading failed')
            self.d = {}
            self.num = 0
            self.article_set = set()

    def _read_user_agent(self):
        with open('user_agents.txt', 'r') as f:
            data = []
            for l in f.readlines():
                data.append(''.join(l.split('\n')))
            return data

    def _get_content(self, url):
        user_agent = random.choice(self.user_agent)
        headers = {'user-agent': user_agent}
        if not self.silent:
            print('url:', url)
        req = requests.get(url=url, headers=headers, timeout=(2, 5))
        return req

    def get_links(self, req, url):
        try:
            url_cmpnts = urlsplit(url)
            path = url_cmpnts.path
            sub_paths = re.split(r'/', path)[1:]
            # print(sub_paths)

            content_start_i = 0
            for i, p in enumerate(sub_paths):
                if re.search(r'[0-9]{5,}', p) is not None or re.search(r'-', p) is not None:
                    # print('here')
                    content_start_i = i
                    break
            # print(content_start_i)

            contents = sub_paths[content_start_i:]

            path_pattern = '/'
            for p in sub_paths[:content_start_i]:
                pa = re.sub('[0-9]', '[0-9]', p)
                pa = re.sub('^[a-z]{3}$', '[a-z]{3}', pa)
                path_pattern += pa + '/'

            links = re.findall('href="(.*?' + path_pattern + '.*?)"', req.text)
            links = [l for l in links if re.match('[h/]', l)]
            for i, l in enumerate(links):
                if re.match('/', l):
                    links[i] = urljoin(url, l)
            links = [re.findall(r'(http[^"]*)', l)[0] for l in links if
                     re.search(r'http', l) and not self._is_duplicate(contents, l)]
            links = [l for l in links if re.search(url_cmpnts.netloc, l) and not re.search(r'\.[a-z]{3}$', l)]
            links = [l for l in links if len(re.split(r'/', urlsplit(l).path)[1:]) == len(sub_paths)]

            link_set = set()
            for l in links:
                link_set.add(l)
            links = list(link_set)


        except:
            links = []
        return links

    def _is_duplicate(self, contents, link):
        for c in contents:
            if not re.search(c, link):
                return False
        return True

    def get_text(self, req):
        try:
            text = req.text
            paras = re.findall(r'<p>([^<>]+?)</p>', text)

            article = ''
            for para in paras:
                article += (para + '\n')
            article = self._filter_texts(article)
            return article
        except:
            return None

    def _filter_texts(self, article):
        article = re.sub(r'(?=\.*)\n\n.*', '', article)

        sens = sent_tokenize(article)
        if len(sens) < 2:
            return None
        word_num = sum([len(word_tokenize(sen)) for sen in sens])
        if word_num < 50:
            return None
        return article

    def process(self, url):
        req = self._get_content(url)
        text = self.get_text(req)
        links = self.get_links(req, url)
        if not self.silent:
            print('****************************Get Links**************************')
            for l in links:
                print(l)
        if not self.silent:
            print('****************************Get Text**************************')
            print(text)
        return text, links

    def bfs_crawl(self, *entry_urls, save_name='neg', size_limit=100, num_limit=8000, crawl_time=10, ):
        assert len(entry_urls) >= 1, 'Must exist one url!'
        Q = deque()
        S = set()
        for url in entry_urls:
            Q.append(url)

        start = time.time()
        num = self.num
        size = 0
        order = 0
        retry_count = 0
        retry_total = 5
        while True:
            url = Q.popleft()
            # print(url)
            order += 1
            try:
                text, links = self.process(url)
                retry_count = 0
                if text and text not in self.article_set:
                    num += 1
                    self.article_set.add(text)
                    size = self.save_function(save_name, text)
                    crawling_time = time.time() - start
                    size_c = size > size_limit
                    time_c = (crawling_time > crawl_time * 60 * 60)
                    num_c = num > num_limit
                    if size_c or time_c or num_c:
                        break
                S.add(url)
                for link in links:
                    if link not in S:
                        Q.append(link)
            except:
                retry_count += 1
                if retry_count == retry_total:
                    if not self.silent:
                        print('Retrying {} times failed, exiting...'.format(retry_count))
                    retry_count = 0
                else:
                    if not self.silent:
                        print('Retrying {}/{} times'.format(retry_count, retry_total))
                    Q.appendleft(url)

            if self.silent:
                crawling_time = time.time() - start
                sys.stdout.write(
                    "\rCrawling{:6} Num:{}, Size:{}, Time:{}, Queus:{}".format('.' * (order % 6), num, round(size, 2),
                                                                     round(crawling_time, 2), len(Q)))
                sys.stdout.flush()

            time.sleep(0.1)

    def save_function(self, save_name, text):
        name = 'train-'
        num = len(self.d)
        self.d[name + str(num)] = {'text': text, 'label': 0}
        with open(save_name + '.json', 'w') as f:
            json.dump(self.d, f)
        return os.path.getsize(save_name + '.json') / 1024 / 1024


class GoogleCrawler():
    def __init__(self):
        self.domains = self._read_domaines()
        self.user_agent = self._read_user_agent()
        self.conf = self._read_config()

    def _read_domaines(self):
        with open('domine.txt', 'r') as f:
            data = []
            for l in f.readlines():
                data.append(''.join(l.split('\n')))
            return data

    def _read_user_agent(self):
        with open('user_agents.txt', 'r') as f:
            data = []
            for l in f.readlines():
                data.append(''.join(l.split('\n')))
            return data

    def _read_config(self):
        if os.path.exists('Config.ini'):
            conf = {}
            print('[~] Load config file')
            config = configparser.ConfigParser()
            config.read("Config.ini", encoding='utf-8')
            conf['proxy'] = config['config']['proxy']
            conf['link'] = int(config['config']['link'])
            conf['page'] = int(config['config']['page'])
            conf['sleep'] = int(config['config']['sleep'])
            conf['halt'] = int(config['config']['halt'])
            conf['using proxy'] = int(config['config']['using proxy'])
            conf['auto proxy'] = int(config['config']['auto proxy'])
            conf['timeout'] = int(config['config']['timeout'])
            conf['retry'] = int(config['config']['retry'])

            print('[+] Load finished')
            return conf
        else:
            print('[-] Cannot find config')
            exit()

    def _search(self, query, page):
        domain = random.choice(self.domains)
        user_agent = random.choice(self.user_agent)
        headers = {'user-agent': user_agent}
        conf = self.conf

        auto_proxy = conf['auto proxy']
        using_proxy = conf['using proxy']
        timeout = conf['timeout']
        if using_proxy == 1:
            if auto_proxy == 0:
                proxy = conf['proxy']
            else:
                proxy = None
            proxies = {'http': 'http://{}'.format(proxy), 'https': 'https://{}'.format(proxy)}
        else:
            proxies = None
        url = 'https://{}/search?q={}&start={}'.format(domain, query, page)

        # print('url:', url)
        req = requests.get(url=url, headers=headers, proxies=proxies, timeout=timeout)
        return req

    def _process_req(self, req):
        conf = self.conf
        bt = BeautifulSoup(req.content, 'lxml')
        anchors = bt.find_all('a')
        links = [a.attrs['href'] for a in anchors]
        links = [re.sub(r'[&%].+', '', l) for l in links]
        links = [l for l in links if re.search(r'http', l) and not re.search(r'google', l)]
        links = [re.findall(r'(http.+)', l)[0] for l in links]
        order_dict = {}
        order = 0
        for l in links:
            order_dict[l] = order
            order += 1
        order_tuple = sorted(order_dict.items(), key=lambda x: x[1])
        links = [l for l, _ in order_tuple]
        num = conf['link']
        links = links[:num]
        # print(links)
        assert len(links) != 0
        return links

    def _process_query(self, query):
        # query = re.split(r'\n', query)[0]
        try:
            sens = sent_tokenize(query)
            q = sens[0] + ' ' + sens[1]
            q = re.sub(r'[^\w]+', ' ', q)
        except:
            q = '{' + query + '}'
        return q

    def query_links(self, topic, links_num):
        query = self._process_query(topic)
        conf = self.conf
        sleeptime = conf['sleep']
        halttime = conf['halt']
        retry = conf['retry']

        links = set()

        page = 0
        while True:

            times = 0
            req = self._search(query, page)
            while True:
                if b'302 Moved' not in req.content:
                    try:
                        link = self._process_req(req)
                        for l in link:
                            links.add(l)
                        print(len(links))
                        if len(links) >= links_num:
                            return links
                        times = retry
                        page += 10

                        break
                    except:
                        print('Query failed')
                        print('Retrying after:{}sec, {}/{} times'.format(sleeptime, times + 1, retry))
                        times += 1
                        if times == retry:
                            print('Stop retrying')
                            break
                    time.sleep(sleeptime)
                else:
                    print('Rejected by Google')
                    time.sleep(halttime)
        return links


