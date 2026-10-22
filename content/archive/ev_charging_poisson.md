Title: Waiting for the Station
Date: 2026-04-13
Category: Blog
Tags: statistics, EV, charging
Status: draft

As my friend and I sat idle in line to charge the car in front of a Walmart in Mountain View, CA, I looked over at the time and wondered how long I would have to keep waiting here until a stall would finally open up. It was past 6:30pm and the rush of EV owners and Walmart shoppers filled the parking lot to the brim as the line to charge streched back towards the main street. Every charging stall was filled and we didn't know how long we would be waiting. We just waited. With time on our hands, we began to think if there was any way we could estimate how long we would be stuck here, waiting in line behind a Hyundai and a string of shiney EVs lining the outside edge of the parking lot. I think we came up with some good ideas!

## Waiting for the Charger is a Poisson Process

In general, we assumed we did not know how long people had been waiting in line in front of us. Practically, we could tell who had been waiting around the longest, but it makes the problem a little more difficult. So because of this, we wait until someone in line ahead of us gets an open stall. From there, we begin the count.

