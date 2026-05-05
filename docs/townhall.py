# -*- coding: utf-8 -*-
import scrapy
from items import CommentItem
import textblob
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

class TownhallSpider(scrapy.Spider):
    name = 'townhall'
    allowed_domains = ['townhall.virginia.gov']
    start_urls = ['https://www.townhall.virginia.gov/L/comments.cfm?GDocForumID=452']
    custom_settings = {
        'FEED_EXPORTERS' : {
            "jsonlines": "scrapy.exporters.JsonLinesItemExporter",
        },
        'FEED_URI' : '%(name)s_%(time)s.json',
        'FEED_FORMAT': 'jsonlines'
    }

    def parse(self, response):
        rows = response.css('#contentwide>table>tr')
        # cut out the header row
        for each in rows[1:]:
        # for each in rows[1:6]:
            cols = each.xpath('.//td')
            linkfollow = cols[0].css('a::attr(href)').get()
            comment_title = cols[0].xpath('a/text()').get()
            # clean up
            commenter = cols[1].xpath('text()').get()
            # clean up
            date = cols[2].xpath('a/text()').get()
            print(f'{comment_title}  |  {commenter}')
            yield response.follow(linkfollow, callback = self.parse_comment)

    def parse_comment(self, response):
        entry = CommentItem()
        text = response.css('.divComment>p::text').get()
        text = text.replace(u'\u00a0',' ')
        entry['comment'] = text
        blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
        entry['sentiment'] = blob.sentiment.classification
        entry['sentiment_pos'] = blob.sentiment.p_pos
        entry['sentiment_neg'] = blob.sentiment.p_neg
        # yield CommentItem(comment = response.css('.divComment>p::text').get())
        yield entry
