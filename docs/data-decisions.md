# Data preparation decisions

## Forecast target
Weekly observed units sold for physical products.
Returns are not subtracted from this target.

## Sheet overlap
Use the first sheet before 2010-12-01.
Use the second sheet from 2010-12-01 onward.

## Sales selection
Keep positive quantities and positive prices.
Exclude cancellations and reviewed out-of-scope stock codes.
Retain missing customer IDs and descriptions.

## Repeated rows
Retain exact repeated rows initially because the source has no
unique transaction-line identifier to establish accidental duplication.

There are 11,731 extra identical rows after merchandise filtering.
Compare results with and without exact-row deduplication later.

## Weekly coverage

There are 104 complete calendar weeks in the selected date range.

Two weeks contain no source transaction records:
- 2009-12-28
- 2010-12-27

The reason is unconfirmed. Treat sales in these weeks as unknown,
not zero. Preserve their positions in the weekly calendar.

Transactions in other weeks do not prove complete source coverage.