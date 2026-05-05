import scrapy


class ForumItem(scrapy.Item):
    forum_id  = scrapy.Field()
    reg_title = scrapy.Field()
    reg_desc  = scrapy.Field()


class CommentItem(scrapy.Item):
    forum_id   = scrapy.Field()
    comment_id = scrapy.Field()
    author     = scrapy.Field()
    date       = scrapy.Field()
    title      = scrapy.Field()
    text       = scrapy.Field()
