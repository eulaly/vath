
# Table of Contents

1.  [Project Goals](#org863a759)
2.  [Architecture](#orgcd91fd0)
    1.  [Scraper](#org3256ad3)
    2.  [Storage](#org7a9a92c)
    3.  [Analysis](#org6ed72dc)
3.  [Roadmap](#org416f14d)



<a id="org863a759"></a>

# Project Goals

1.  Document and analyze sentiment of public comments on Virginia law, to determine:
    1.  the utility of this forum as a mechanism for public comment, and
    2.  the impact of this forum on Virginia regulation.
2.  Make data and insights broadly available.
3.  Generalize to other public comment tools.


<a id="orgcd91fd0"></a>

# Architecture

1.  Scrape/Parse: ****Scrapy**** for downloading comments
2.  Storage: json
3.  Sentiment analysis: Claude haiku
4.  Display: TBD


<a id="org3256ad3"></a>

## Scraper

Scrapy provides a simple mechanism for browsing and 

1.  Forums listing page: \`Forums.cfm\` - lists all open forums with agency, reg title, action type, brief description, closing date, comment count
2.  Comment listing page: \`comments.cfm?GDocForumID=X\` or \`comments.cfm?stageid=X\` or \`comments.cfm?petitionid=X\` - lists comments with title, author, date
3.  Individual comment page: \`viewcomments.cfm?commentid=X\` - shows regulation title + brief description at the top, plus the comment


<a id="org7a9a92c"></a>

## Storage

One JSONL file per forum/bill.


<a id="org6ed72dc"></a>

## Analysis

Google and Amazon both return generic sentiment (tone of writing: positive/negative), not stance (for/against the regulation): "I strongly believe the government should NOT interfere" is negative tone but "against" the regulation.  We will run the forum/bill title and cache the entirety of the proposed change, perhaps as a fallback.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Tool</th>
<th scope="col" class="org-left">Output</th>
<th scope="col" class="org-left">Context</th>
<th scope="col" class="org-left">Sarcasm</th>
<th scope="col" class="org-left">Context window</th>
<th scope="col" class="org-left">Cost/1k comments</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">Google NL API</td>
<td class="org-left">-1→+1, magnitude</td>
<td class="org-left">No/generic</td>
<td class="org-left">Poorly</td>
<td class="org-left">No</td>
<td class="org-left">~$1–2</td>
</tr>

<tr>
<td class="org-left">Amazon Comprehend</td>
<td class="org-left">Pos/Neg/Neutral/Mixed</td>
<td class="org-left">No/generic</td>
<td class="org-left">Poorly</td>
<td class="org-left">No</td>
<td class="org-left">~$0.10</td>
</tr>

<tr>
<td class="org-left">Claude Haiku</td>
<td class="org-left">Prompted → for/against/neutral</td>
<td class="org-left">Yes</td>
<td class="org-left">Yes, with prompt</td>
<td class="org-left">Yes</td>
<td class="org-left">~$0.10–0.30</td>
</tr>

<tr>
<td class="org-left">GPT-4o-mini</td>
<td class="org-left">Prompted → same</td>
<td class="org-left">Yes</td>
<td class="org-left">Yes</td>
<td class="org-left">Yes</td>
<td class="org-left">~$0.05–0.15</td>
</tr>
</tbody>
</table>


<a id="org416f14d"></a>

# Roadmap

1.  Scrape one forum
2.  Compare sentiment models
3.  Display
4.  Scrape all data
5.  Scale?

