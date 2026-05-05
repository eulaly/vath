"""Tests for ForumSpider parsing logic using fake HTML responses."""

from scrapy.http import HtmlResponse, Request

from scraper.spiders.forum import ForumSpider


def fake_response(url, body, meta=None):
    req = Request(url=url, meta=meta or {})
    return HtmlResponse(url=url, body=body.encode("utf-8"), request=req)


# ---------------------------------------------------------------------------
# Minimal page HTML fragments

PAGE1_HTML = """
<html><body>
  <strong>Guidance Document Change:</strong> The Model Policies for the Treatment of Transgender Students
  was developed in response to House Bill 145 and Senate Bill 161.

  <div style="font-family: verdana;">
    <form name="page" id="page" action="ViewComments.cfm?GdocForumID=452" method="post">
      <input name="vPage" id="vpage" type="input" value="1">
      <input name="vPerPage" id="vPerPage" type="input" value="500">
      <a href="javascript:document.page.vpage.value=3;document.page.submit();">3</a>
      <a href="javascript:document.page.vpage.value=2;document.page.submit();">Next</a>
      <input type="submit" name="sub1" value="go">
    </form>
  </div>

  <div id="cbox101" class="Cbox">
    <div style="float: right; text-align: right;">
      <div style="background-color: white; border: 1px solid #cccccc; padding: 4px">1/4/21&nbsp;&nbsp;9:15 am</div>
    </div>
    <div>
      <strong>Commenter:</strong>
      Alice Example
      <br><br>
      <strong>I strongly support this</strong>
    </div>
    <div style="clear: right">&nbsp;</div>
    <div class="divComment">
      <p>This is a great policy for students.</p>
      <p>All schools should follow it.</p>
    </div>
    <div style="float: left; font-size: 90%;">
      CommentID: <a href="ViewComments.cfm?commentid=101">101</a>
    </div>
  </div>

  <div id="cbox102" class="Cbox">
    <div style="float: right; text-align: right;">
      <div style="background-color: white; border: 1px solid #cccccc; padding: 4px">1/5/21&nbsp;&nbsp;10:00 am</div>
    </div>
    <div>
      <strong>Commenter:</strong>
      Bob Sample
      <br><br>
      <strong>Opposed</strong>
    </div>
    <div style="clear: right">&nbsp;</div>
    <div class="divComment">
      <p>I do not support this guidance.</p>
    </div>
    <div style="float: left; font-size: 90%;">
      CommentID: <a href="ViewComments.cfm?commentid=102">102</a>
    </div>
  </div>
</body></html>
"""

PAGE2_HTML = """
<html><body>
  <div id="cbox201" class="Cbox">
    <div style="float: right; text-align: right;">
      <div style="background-color: white; border: 1px solid #cccccc; padding: 4px">1/6/21&nbsp;&nbsp;11:00 am</div>
    </div>
    <div>
      <strong>Commenter:</strong>
      Carol T
      <br><br>
      <strong>Support</strong>
    </div>
    <div style="clear: right">&nbsp;</div>
    <div class="divComment">
      <p>This policy is long overdue.</p>
    </div>
  </div>
</body></html>
"""


def make_spider():
    return ForumSpider()


# ---------------------------------------------------------------------------

def test_page1_generates_remaining_page_requests():
    spider = make_spider()
    response = fake_response(
        "https://www.townhall.virginia.gov/L/ViewComments.cfm?GdocForumID=452",
        PAGE1_HTML,
        meta={"is_first": True},
    )
    results = list(spider.parse_comments(response))
    form_reqs = [r for r in results if isinstance(r, scrapy.FormRequest)]
    # Pages 2 and 3 should be requested (last page link = 3)
    assert len(form_reqs) == 2
    pages = sorted(r.body.decode() for r in form_reqs)
    assert "vPage=2" in pages[0]
    assert "vPage=3" in pages[1]


def test_page1_yields_items():
    spider = make_spider()
    response = fake_response(
        "https://www.townhall.virginia.gov/L/ViewComments.cfm?GdocForumID=452",
        PAGE1_HTML,
        meta={"is_first": True},
    )
    results = list(spider.parse_comments(response))
    from scraper.items import CommentItem
    items = [r for r in results if isinstance(r, CommentItem)]
    assert len(items) == 2


def test_comment_fields_parsed_correctly():
    spider = make_spider()
    response = fake_response(
        "https://www.townhall.virginia.gov/L/ViewComments.cfm?GdocForumID=452",
        PAGE1_HTML,
        meta={"is_first": True},
    )
    from scraper.items import CommentItem
    items = [r for r in spider.parse_comments(response) if isinstance(r, CommentItem)]
    item = items[0]
    assert item["comment_id"] == "101"
    assert item["author"] == "Alice Example"
    assert item["title"] == "I strongly support this"
    assert "great policy" in item["text"]
    assert "All schools" in item["text"]  # multi-paragraph joined
    assert "1/4/21" in item["date"]


def test_reg_context_extracted():
    spider = make_spider()
    response = fake_response(
        "https://www.townhall.virginia.gov/L/ViewComments.cfm?GdocForumID=452",
        PAGE1_HTML,
        meta={"is_first": True},
    )
    from scraper.items import CommentItem
    items = [r for r in spider.parse_comments(response) if isinstance(r, CommentItem)]
    item = items[0]
    assert "Transgender Students" in item["reg_title"]
    assert "House Bill 145" in item["reg_desc"]


def test_subsequent_page_uses_meta_reg_context():
    spider = make_spider()
    response = fake_response(
        "https://www.townhall.virginia.gov/L/ViewComments.cfm?GdocForumID=452",
        PAGE2_HTML,
        meta={"reg_title": "Test Reg", "reg_desc": "Full description text"},
    )
    from scraper.items import CommentItem
    items = [r for r in spider.parse_comments(response) if isinstance(r, CommentItem)]
    assert len(items) == 1
    assert items[0]["reg_title"] == "Test Reg"
    assert items[0]["author"] == "Carol T"


def test_last_page_detection():
    spider = make_spider()
    response = fake_response(
        "https://www.townhall.virginia.gov/L/ViewComments.cfm?GdocForumID=452",
        PAGE1_HTML,
        meta={"is_first": True},
    )
    assert spider._last_page(response) == 3


import scrapy

SPAN_WRAPPED_HTML = """
<html><body>
  <strong>Guidance Document Change:</strong> Some regulation was developed.

  <form name="page" id="page" action="ViewComments.cfm?GdocForumID=452" method="post">
    <input name="vPage" value="1"><input name="vPerPage" value="500">
    <a href="javascript:document.page.vpage.value=1;document.page.submit();">1</a>
    <input type="submit" name="sub1" value="go">
  </form>

  <div id="cbox301" class="Cbox">
    <div style="float: right; text-align: right;">
      <div style="background-color: white; border: 1px solid #cccccc; padding: 4px">2/1/21&nbsp;&nbsp;8:00 am</div>
    </div>
    <div>
      <strong>Commenter:</strong>
      Dan Span
      <br><br>
      <strong>Opposed</strong>
    </div>
    <div style="clear: right">&nbsp;</div>
    <div class="divComment">
      <!DOCTYPE html><html><head></head><body>
      <p style="margin: 0in;"><span style="font-size: 10.5pt;">Text inside a span element.</span></p>
      </body></html>
    </div>
  </div>
</body></html>
"""


def test_span_wrapped_text_is_extracted():
    spider = make_spider()
    response = fake_response(
        "https://www.townhall.virginia.gov/L/ViewComments.cfm?GdocForumID=452",
        SPAN_WRAPPED_HTML,
        meta={"is_first": True},
    )
    from scraper.items import CommentItem
    items = [r for r in spider.parse_comments(response) if isinstance(r, CommentItem)]
    assert len(items) == 1
    assert "Text inside a span element" in items[0]["text"]
