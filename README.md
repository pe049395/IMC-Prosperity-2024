# IMC-Prosperity-2024

IMC Prosperity is an algorithmic trading competition hosted by IMC Trading. The competition has two parts: algorithmic trading and manual trading. In the algorithmic trading section, new products are introduced each round, allowing us to implement various trading strategies. In the manual trading section, simple math and game theory questions are given. For more information, you can refer to the [official wiki](https://imc-prosperity.notion.site/Prosperity-2-Wiki-fe650c0292ae4cdb94714a3f5aa74c85).

##### Our Team
- Byeongguk Kang
- Minwoo Kim
- Uihyung Lee

|Rank|Overall|Algorithmic|Manual|
|------|---|---|---|
|Round 1|28|28||
|Round 2|2|2||
|Round 3|4|2||
|Round 4|5|3||
|Round 5|13|10||

## Algorithmic Trading
### Round 1
##### Description
The first two tradable products are introduced: `STARFRUIT` and `AMETHYSTS`. While the value of the `AMETHYSTS` has been stable throughout the history of the archipelago, the value of `STARFRUIT` has been going up and down over time.

##### Fair Value
We compared the fair price with the order book. If there was a mispricing, we execute a market order to capture the arbitrage opportunity, while making markets under normal circumstances.

Initially, we aimed to trade around the midprice. However, midprice had noise from traders placing orders at unusual prices. To mitigate this, we tried trading using micro prices (volume-weighted average midprice).

There was a slight difference between the PnL in our backtesting tool and pnl in IMC dashboard. IMC used the hidden fair value for PnL calculation. Additionally, both bid and ask sides consistently had one level with significantly large quantities. We discovered that the average of these two prices closely resembled the hidden fair value. Therefore, we traded around the average of the two prices.

##### Strategy
**Market Making:**
Our goal was to place orders that maximize the expected utility per trade, calculated as `(profit + other utility increment) * execution probability`.

For example, if a buy order at 9998 was executed when the fair value was 10000, the profit from that trade would be 2.

Other utility includes inventory risk. When market making, keeping a position close to zero is highly advantageous. Larger positions involve higher exposure to market risk. And as positions approach their limits, the potential for profitable trades decreases due to smaller execution quantities.

Initially, we implemented Ornstein-Uhlenbeck process-based market making for `AMETHYSTS` and $dS=\sigma dW$ process based market making for `STARFRUIT`.

However, these market making models were not perfectly suited to our market since the execution probability was modeled using an exponential function. To address this, we attempted to estimate the execution probability using a Poisson distribution. Unfortunately, this did not result in a significant improvement in PnL, so we reverted back to the original model.

**Market Taking:**
When there were bid prices higher than the fair value, we executed market sell orders to profit from the mispricing. Also, we executed market sell orders even if the bid price was at the fair price and we had positive positions, to reduce our position.

We followed a similar approach when executing market buy orders.


### Round 2
##### Description
`ORCHIDS` are very delicate and their value is dependent on all sorts of observable factors like hours of sun light, humidity, shipping costs, in- & export tariffs and suitable storage space.

##### Strategy
We couldn't find strong connections between `ORCHIDS` and the various factors. So we didn't do directional bets or price prediction.

Initially, we considered market making due to the wide bid-ask spread of `ORCHIDS`, but found it impossible because bids lower than the fair price and asks higher than the fair price did not fill well.

Instead, `ORCHIDS` were also tradable on the exchange of the south archipelago. Therefore, we implemented arbitrage between the two exchanges. We placed orders to maximize expected profit per trade (= `(enter price on this island - exit price on the south archipelago - shipping cost - import tariff) * execution probability`). We estimated the execution probability through our own experiment.
