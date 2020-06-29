from crawlers import GoogleCrawler, ArticleCrawler


def claw():
    gc = GoogleCrawler()
    links = gc.query_links('climate change', 130)
    ac = ArticleCrawler()
    ac.bfs_crawl(*links)



if __name__ == '__main__':
    claw()


