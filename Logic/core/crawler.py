from requests import get
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json
import time


class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0',
        'Accept-Language': 'en-US',
    }
    top_250_URL = 'https://www.imdb.com/chart/top/'
    BASE_MOVIE_URL = 'https://www.imdb.com/title/'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        self.crawling_threshold = crawling_threshold
        self.not_crawled = []
        self.crawled = []
        self.added_ids = set()
        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()

    def get_soup(self, url):
        response = self.crawl(url)
        if response.status_code != 200:
            print(f'Error in getting {url}: {response}')
        else:
            return BeautifulSoup(response.text, 'html.parser')

    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        return URL.split('/')[4]

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        with open('IMDB_crawled.json', 'w') as f:
            f.write(json.dumps(self.crawled))
            f.close()
        with open('IMDB_not_crawled.json', 'w') as f:
            f.write(json.dumps(self.not_crawled))
            f.close()

    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        with open('IMDB_crawled.json', 'r') as f:
            crawled_data = f.read()
            f.close()
            self.crawled = json.loads(crawled_data)

        with open('IMDB_not_crawled.json', 'r') as f:
            not_crawled_data = f.read()
            f.close()
            self.not_crawled = json.loads(not_crawled_data)

        self.added_ids = [movie_info['id'] for movie_info in self.crawled]

    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        return get(URL, headers=self.headers)

    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        top_250_soup = self.get_soup(self.top_250_URL)
        anchors = top_250_soup.select('a[href]')
        for anchor in anchors:
            if anchor['href'].startswith('/title'):
                movie_id = anchor['href'].split('/')[2]
                url = self.BASE_MOVIE_URL + movie_id
                if url not in self.not_crawled:
                    self.not_crawled.append(url)

    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """
        # Populate not_crawled if empty
        if len(self.not_crawled) == 0:
            self.extract_top_250()

        futures = []
        crawled_counter = 0

        with ThreadPoolExecutor(max_workers=10) as executor:
            while len(self.crawled) < self.crawling_threshold:
                self.add_queue_lock.acquire()
                URL = self.not_crawled.pop(0)
                self.add_queue_lock.release()
                futures.append(executor.submit(self.crawl_page_info, URL))
                crawled_counter += 1
                if crawled_counter % 10 == 0:
                    print(f"Crawled {crawled_counter}")
                if len(self.not_crawled) == 0:
                    wait(futures)
                    futures = []

    def crawl_page_info(self, URL):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.
        
        Parameters
        ----------
        URL: str
            The URL of the site
        """
        movie_id = self.get_id_from_URL(URL)
        if movie_id in self.added_ids:
            return

        print(f"new iteration id={movie_id}")

        soup = self.get_soup(URL)
        time.sleep(1)
        mpaa_soup = self.get_soup(self.get_mpaa_link(URL))
        time.sleep(1)
        summary_soup = self.get_soup(self.get_summary_link(URL))
        time.sleep(1)
        review_soup = self.get_soup(self.get_review_link(URL))

        if soup is not None and mpaa_soup is not None and summary_soup is not None and review_soup is not None:
            movie = self.extract_movie_info(
                soup=soup,
                mpaa_soup=mpaa_soup,
                summary_soup=summary_soup,
                review_soup=review_soup,
                movie=self.get_imdb_instance(),
                URL=URL,
            )
            # Add movie to crawled data & it's id to added ids
            self.add_list_lock.acquire()

            if movie_id not in self.added_ids:
                self.crawled.append(movie)
                self.added_ids.add(movie_id)

            # Add related movies url to not crawled list
            self.add_queue_lock.acquire()
            for related_link in movie['related_links']:
                related_id = self.get_id_from_URL(related_link)
                if related_id not in self.added_ids and related_link not in self.not_crawled:
                    self.not_crawled.append(related_link)
            self.add_queue_lock.release()

            self.add_list_lock.release()
        else:
            # Add failed crawl url to not_crawled
            self.add_queue_lock.acquire()
            self.not_crawled.append(URL)
            self.add_queue_lock.release()
            print(f"crawl failed for id={movie_id}")

    def extract_movie_info(self, soup, mpaa_soup, summary_soup, review_soup, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        soup: bs4.BeautifulSoup
            The soup of get response
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        movie['id'] = self.get_id_from_URL(URL)
        movie['title'] = self.get_title(soup)
        movie['first_page_summary'] = self.get_first_page_summary(soup)
        movie['release_year'] = self.get_release_year(soup)
        movie['budget'] = self.get_budget(soup)
        movie['gross_worldwide'] = self.get_gross_worldwide(soup)
        movie['directors'] = self.get_director(soup)
        movie['writers'] = self.get_writers(soup)
        movie['stars'] = self.get_stars(soup)
        movie['related_links'] = self.get_related_links(soup)
        movie['genres'] = self.get_genres(soup)
        movie['languages'] = self.get_languages(soup)
        movie['countries_of_origin'] = self.get_countries_of_origin(soup)
        movie['rating'] = self.get_rating(soup)

        movie['mpaa'] = self.get_mpaa(mpaa_soup)

        movie['summaries'] = self.get_summary(summary_soup)
        movie['synopsis'] = self.get_synopsis(summary_soup)

        movie['reviews'] = self.get_reviews_with_scores(review_soup)

        return movie

    def get_mpaa_link(self, url):
        return url + '/parentalguide'

    def get_summary_link(self, url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            return url + '/plotsummary'
        except:
            print("failed to get summary link")
            return None

    def get_review_link(self, url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            return url + '/reviews'
        except:
            print("failed to get review link")
            return None

    def get_title(self, soup):
        """
        Get the title of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """
        try:
            return soup.find('span', class_='hero__primary-text').text
        except:
            print("failed to get title")
            return ''

    def get_first_page_summary(self, soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        try:
            return soup.find('span', class_='sc-466bb6c-0').text
        except:
            print("failed to get first page summary")
            return ''

    def get_related_metadata_list_items(self, soup, related_names):
        related_items = []
        list_items = soup.find_all(class_='ipc-metadata-list__item')
        for item in list_items:
            item_labels = item.find_all(class_='ipc-metadata-list-item__label')
            for label in item_labels:
                try:
                    if label.text in related_names:
                        related_items.append(item)
                except:
                    continue
        return related_items

    def get_director(self, soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        directors = []
        try:
            director_list_items = self.get_related_metadata_list_items(
                soup=soup,
                related_names=['Director', 'Directors'],
            )
            for item in director_list_items:
                names = item.find_all(class_='ipc-metadata-list-item__list-content-item')
                for name in names:
                    if name.text not in directors:
                        directors.append(name.text)
        except:
            print("failed to get director")
        return directors

    def get_stars(self, soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        stars = []
        try:
            stars_list_items = self.get_related_metadata_list_items(
                soup=soup,
                related_names=['Star', 'Stars'],
            )
            for item in stars_list_items:
                names = item.find_all(class_='ipc-metadata-list-item__list-content-item')
                for name in names:
                    if name.text not in stars:
                        stars.append(name.text)
        except:
            print("failed to get stars")
        return stars

    def get_writers(self, soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        writers = []
        try:
            writers_list_items = self.get_related_metadata_list_items(
                soup=soup,
                related_names=['Writer', 'Writers'],
            )
            for item in writers_list_items:
                names = item.find_all(class_='ipc-metadata-list-item__list-content-item')
                for name in names:
                    if name.text not in writers:
                        writers.append(name.text)
        except:
            print("failed to get writers")
        return writers

    def get_related_links(self, soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        related_links = []
        try:
            sections = soup.find_all('section', class_='ipc-page-section ipc-page-section--base celwidget')
            for section in sections:
                spans = section.find_all('span')
                for span in spans:
                    if span.text == 'More like this':
                        anchors = section.find_all('a', class_='ipc-lockup-overlay ipc-focusable')
                        for anchor in anchors:
                            movie_id = anchor['href'].split('/')[2]
                            movie_link = self.BASE_MOVIE_URL + movie_id
                            if movie_link not in related_links:
                                related_links.append(movie_link)
        except:
            print("failed to get related links")
        return related_links

    def get_related_sections(self, soup, related_name):
        related_sections = []
        sections = soup.find_all('section', class_='ipc-page-section')
        for section in sections:
            spans = section.find_all('span')
            for span in spans:
                try:
                    if span.text == related_name:
                        related_sections.append(section)
                except:
                    continue
        return related_sections

    def get_summary(self, soup):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        summary_list = []
        try:
            summary_sections = self.get_related_sections(soup, 'Summaries')
            for section in summary_sections:
                summaries = section.find_all('div', class_='ipc-html-content-inner-div')
                for summary in summaries:
                    summary_list.append(summary.text)
        except:
            print("failed to get summary")
        return summary_list

    def get_synopsis(self, soup):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        synopsis_list = []
        try:
            synopsis_sections = self.get_related_sections(soup, 'Synopsis')
            for section in synopsis_sections:
                synopsis = section.find_all('div', class_='ipc-html-content-inner-div')
                for synop in synopsis:
                    synopsis_list.append(synop.text)
        except:
            print("failed to get synopsis")
        return synopsis_list

    def get_reviews_with_scores(self, soup):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        reviews_list = []
        try:
            review_divs = soup.find_all('div', class_='review-container')
            for div in review_divs:
                score_span = div.find('span', class_='rating-other-user-rating')
                if score_span is not None:
                    score = score_span.find('span').text
                else:
                    score = ''
                review_text = div.find('div', class_='show-more__control').text
                reviews_list.append([review_text, score])
        except:
            print("failed to get reviews")
        return reviews_list

    def get_genres(self, soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        genre_list = []
        try:
            genre_spans = soup.find_all('span', class_='ipc-chip__text')
            for span in genre_spans:
                genre_text = span.text
                if not genre_text.startswith('Back'):
                    genre_list.append(genre_text)
        except:
            print("Failed to get generes")
        return genre_list

    def get_rating(self, soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        rating = ''
        try:
            rating_parent_divs = soup.find_all('div', class_='sc-acdbf0f3-0 haeNPA rating-bar__base-button')
            for parent_div in rating_parent_divs:
                rating_divs = parent_div.find_all('div', class_='sc-acdbf0f3-1 kCTJoV')
                for div in rating_divs:
                    if div.text == 'IMDb RATING':
                        rating_spans = parent_div.find_all('span', class_='sc-bde20123-1 cMEQkK')
                        for span in rating_spans:
                            rating = span.text
                            break
        except:
            print("failed to get rating")
        return rating

    def get_mpaa(self, soup):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        MPAA = ''
        try:
            mpaa = soup.find_all('td')
            if mpaa[0].text == 'MPAA':
                MPAA = mpaa[1].text
        except:
            print("failed to get mpaa")
        return MPAA

    def get_release_year(self, soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            element = soup.find(class_='sc-67fa2588-0')
            return element.find(class_='ipc-link').text
        except:
            print("failed to get release year")
            return ''

    def get_related_list_items_from_list(self, soup, related_names, classname):
        related_table_items = []
        table_items = soup.find_all('li', class_=classname)
        for item in table_items:
            spans = item.find_all('span', class_='ipc-metadata-list-item__label')
            for span in spans:
                try:
                    if span.text in related_names:
                        related_table_items.append(item)
                except:
                    continue
        return related_table_items

    def get_languages(self, soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        languages = []
        try:
            language_list_items = self.get_related_list_items_from_list(
                soup=soup,
                related_names=['Language', 'Languages'],
                classname='ipc-metadata-list__item',
            )
            for item in language_list_items:
                anchors = item.find_all(
                    'a',
                    class_='ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link',
                )
                for anchor in anchors:
                    language = anchor.text
                    if language not in languages:
                        languages.append(language)
        except:
            print("failed to get languages")
        return languages

    def get_countries_of_origin(self, soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        countries_of_origin = []
        try:
            countries_list_items = self.get_related_list_items_from_list(
                soup=soup,
                related_names=['Country of origin', 'Countries of origin'],
                classname='ipc-metadata-list__item',
            )
            for item in countries_list_items:
                anchors = item.find_all(
                    'a',
                    class_='ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link',
                )
                for anchor in anchors:
                    country = anchor.text
                    if country not in countries_of_origin:
                        countries_of_origin.append(country)
        except:
            print("failed to get countries of origin")
        return countries_of_origin

    def get_budget(self, soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        budget = ''
        try:
            budget_list_items = self.get_related_list_items_from_list(
                soup=soup,
                related_names=['Budget'],
                classname='ipc-metadata-list__item sc-1bec5ca1-2 bGsDqT',
            )
            for item in budget_list_items:
                budget_spans = item.find_all('span', class_='ipc-metadata-list-item__list-content-item')
                for span in budget_spans:
                    budget = span.text
        except:
            print("failed to get budget")
        return budget

    def get_gross_worldwide(self, soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        gross = ''
        try:
            gross_list_items = self.get_related_list_items_from_list(
                soup=soup,
                related_names=['Gross worldwide'],
                classname='ipc-metadata-list__item sc-1bec5ca1-2 bGsDqT',
            )
            for item in gross_list_items:
                gross_spans = item.find_all('span', class_='ipc-metadata-list-item__list-content-item')
                for span in gross_spans:
                    gross = span.text
        except:
            print("failed to get gross worldwide")
        return gross


def main():
    imdb_crawler = IMDbCrawler(crawling_threshold=1000)
    imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling()
    imdb_crawler.write_to_file_as_json()


if __name__ == '__main__':
    main()
