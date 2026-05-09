import re
from datetime import datetime

import scrapy

from scraper.items import CommentItem, ForumItem

_BASE = "https://www.townhall.virginia.gov/L/ViewComments.cfm"
_NBSP = "\xa0"
_REPLACEMENT_CHAR = "�"


def _view_url(forum_id):
    return f"{_BASE}?GdocForumID={forum_id}"


def _parse_date(raw):
    normalized = " ".join(raw.split()).upper()
    try:
        return datetime.strptime(normalized, "%m/%d/%y %I:%M %p").isoformat()
    except ValueError:
        return raw


class ForumSpider(scrapy.Spider):
    name = "forum"
    allowed_domains = ["townhall.virginia.gov"]

    # Override at runtime: scrapy crawl forum -a forum_id=452
    forum_id = "452"
    per_page = 500

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.settings.set(
            "FEEDS",
            {
                f"output/forum{spider.forum_id}_comments_%(time)s.jsonl": {
                    "format": "jsonlines",
                    "encoding": "utf-8",
                    "overwrite": False,
                }
            },
            priority="spider",
        )
        return spider

    async def start(self):
        yield scrapy.FormRequest(
            _view_url(self.forum_id),
            formdata={"vPage": "1", "vPerPage": str(self.per_page), "sub1": "go"},
            callback=self.parse_comments,
            meta={"is_first": True},
        )

    # ------------------------------------------------------------------
    def parse_comments(self, response):
        if response.meta.get("is_first"):
            reg_title, reg_desc = self._reg_context(response)
            last_page = self._last_page(response)
            yield ForumItem(
                forum_id=self.forum_id,
                reg_title=reg_title,
                reg_desc=reg_desc,
                scraped_at=datetime.utcnow().isoformat(),
                forum_url=_view_url(self.forum_id),
            )
            for page in range(2, last_page + 1):
                yield scrapy.FormRequest(
                    _view_url(self.forum_id),
                    formdata={"vPage": str(page), "vPerPage": str(self.per_page), "sub1": "go"},
                    callback=self.parse_comments,
                )

        for box in response.css("div.Cbox"):
            yield self._parse_box(box)

    # ------------------------------------------------------------------
    def _parse_box(self, box):
        cbox_id = box.attrib.get("id", "")
        comment_id = cbox_id[len("cbox"):] if cbox_id.startswith("cbox") else ""

        date_raw = (
            box.css("div[style*='float: right'] div::text").get("")
            .replace(_NBSP, " ").strip()
        )

        author = (
            box.xpath('.//strong[contains(text(),"Commenter:")]/following-sibling::text()[1]')
            .get("").strip()
        )

        # Second <strong> in the commenter block is the comment title
        strongs = box.css("div > strong::text").getall()
        title = strongs[-1].strip() if len(strongs) > 1 else ""

        paragraphs = box.css(".divComment *::text, .divComment::text").getall()
        text = " ".join(p.strip() for p in paragraphs if p.strip())
        text = text.replace(_NBSP, " ").replace(_REPLACEMENT_CHAR, "'").strip()

        return CommentItem(
            forum_id=self.forum_id,
            comment_id=comment_id,
            author=author,
            date=_parse_date(date_raw),
            title=title,
            text=text,
        )

    # ------------------------------------------------------------------
    def _reg_context(self, response):
        # Page shows: <strong>Guidance Document Change:</strong> description text...
        label_node = response.xpath('//strong[contains(text(),"Change:")]')

        # Collect all sibling text nodes following the label
        siblings = label_node.xpath("following-sibling::text()").getall()
        raw = " ".join(t.strip() for t in siblings if t.strip())
        raw = raw.replace(_NBSP, " ").replace(_REPLACEMENT_CHAR, "'").strip()

        reg_desc = raw

        # reg_title: text up to the first "was " clause or first 200 chars
        m = re.match(r"^(.+?)\s+(?:was |has |guidance document)", raw, re.IGNORECASE)
        reg_title = m.group(1).strip() if m else raw[:200]

        return reg_title, reg_desc

    def _last_page(self, response):
        hrefs = response.xpath(
            '//form[@name="page"]//a[contains(@href,"vpage.value=")]/@href'
        ).getall()
        pages = [
            int(m.group(1))
            for h in hrefs
            if (m := re.search(r"vpage\.value=(\d+)", h))
        ]
        return max(pages) if pages else 1
