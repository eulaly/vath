import scrapy


class CommentItem(scrapy.Item):
    # Forum / regulation context
    forum_id   = scrapy.Field()
    reg_title  = scrapy.Field()
    reg_desc   = scrapy.Field()

    # Comment metadata
    comment_id = scrapy.Field()
    author     = scrapy.Field()
    date       = scrapy.Field()
    title      = scrapy.Field()

    # Comment content
    text       = scrapy.Field()
