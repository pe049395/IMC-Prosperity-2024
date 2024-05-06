import json
import numpy as np
import math
from statistics import NormalDist
from datamodel import *
from typing import Any

INF = 1e9
normalDist = NormalDist(0,1)


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()


class Status:

    _position_limit = {
        "AMETHYSTS": 20,
        "STARFRUIT": 20,
        "ORCHIDS": 100,
        "CHOCOLATE": 250,
        "STRAWBERRIES": 350,
        "ROSES": 60,
        "GIFT_BASKET": 60,
        "COCONUT": 300,
        "COCONUT_COUPON": 600,
    }

    _state = None

    _realtime_position = {key:0 for key in _position_limit.keys()}

    _hist_order_depths = {
        product:{
            'bidprc1': [],
            'bidamt1': [],
            'bidprc2': [],
            'bidamt2': [],
            'bidprc3': [],
            'bidamt3': [],
            'askprc1': [],
            'askamt1': [],
            'askprc2': [],
            'askamt2': [],
            'askprc3': [],
            'askamt3': [],
        } for product in _position_limit.keys()
    }

    _hist_observation = {
        'sunlight': [],
        'humidity': [],
        'transportFees': [],
        'exportTariff': [],
        'importTariff': [],
        'bidPrice': [],
        'askPrice': [],
    }

    _num_data = 0

    def __init__(self, product: str) -> None:
        """Initialize status object.

        Args:
            product (str): product

        """
        self.product = product

    @classmethod
    def cls_update(cls, state: TradingState) -> None:
        """Update trading state.

        Args:
            state (TradingState): trading state

        """
        # Update TradingState
        cls._state = state
        # Update realtime position
        for product, posit in state.position.items():
            cls._realtime_position[product] = posit
        # Update historical order_depths
        for product, orderdepth in state.order_depths.items():
            cnt = 1
            for prc, amt in sorted(orderdepth.sell_orders.items(), reverse=False):
                cls._hist_order_depths[product][f'askamt{cnt}'].append(amt)
                cls._hist_order_depths[product][f'askprc{cnt}'].append(prc)
                cnt += 1
                if cnt == 4:
                    break
            while cnt < 4:
                cls._hist_order_depths[product][f'askprc{cnt}'].append(np.nan)
                cls._hist_order_depths[product][f'askamt{cnt}'].append(np.nan)
                cnt += 1
            cnt = 1
            for prc, amt in sorted(orderdepth.buy_orders.items(), reverse=True):
                cls._hist_order_depths[product][f'bidprc{cnt}'].append(prc)
                cls._hist_order_depths[product][f'bidamt{cnt}'].append(amt)
                cnt += 1
                if cnt == 4:
                    break
            while cnt < 4:
                cls._hist_order_depths[product][f'bidprc{cnt}'].append(np.nan)
                cls._hist_order_depths[product][f'bidamt{cnt}'].append(np.nan)
                cnt += 1
        cls._num_data += 1
        
        # cls._hist_observation['sunlight'].append(state.observations.conversionObservations['ORCHIDS'].sunlight)
        # cls._hist_observation['humidity'].append(state.observations.conversionObservations['ORCHIDS'].humidity)
        # cls._hist_observation['transportFees'].append(state.observations.conversionObservations['ORCHIDS'].transportFees)
        # cls._hist_observation['exportTariff'].append(state.observations.conversionObservations['ORCHIDS'].exportTariff)
        # cls._hist_observation['importTariff'].append(state.observations.conversionObservations['ORCHIDS'].importTariff)
        # cls._hist_observation['bidPrice'].append(state.observations.conversionObservations['ORCHIDS'].bidPrice)
        # cls._hist_observation['askPrice'].append(state.observations.conversionObservations['ORCHIDS'].askPrice)

    def hist_order_depth(self, type: str, depth: int, size) -> np.ndarray:
        """Return historical order depth.

        Args:
            type (str): 'bidprc' or 'bidamt' or 'askprc' or 'askamt'
            depth (int): depth, 1 or 2 or 3
            size (int): size of data

        Returns:
            np.ndarray: historical order depth for given type and depth

        """
        return np.array(self._hist_order_depths[self.product][f'{type}{depth}'][-size:], dtype=np.float32)
    
    @property
    def timestep(self) -> int:
        return self._state.timestamp / 100

    @property
    def position_limit(self) -> int:
        """Return position limit of product.

        Returns:
            int: position limit of product

        """
        return self._position_limit[self.product]

    @property
    def position(self) -> int:
        """Return current position of product.

        Returns:
            int: current position of product

        """
        if self.product in self._state.position:
            return int(self._state.position[self.product])
        else:
            return 0
    
    @property
    def rt_position(self) -> int:
        """Return realtime position.

        Returns:
            int: realtime position

        """
        return self._realtime_position[self.product]

    def _cls_rt_position_update(cls, product, new_position):
        if abs(new_position) <= cls._position_limit[product]:
            cls._realtime_position[product] = new_position
        else:
            raise ValueError("New position exceeds position limit")

    def rt_position_update(self, new_position: int) -> None:
        """Update realtime position.

        Args:
            new_position (int): new position

        """
        self._cls_rt_position_update(self.product, new_position)
    
    @property
    def bids(self) -> list[tuple[int, int]]:
        """Return bid orders.

        Returns:
            dict[int, int].items(): bid orders (prc, amt)

        """
        return list(self._state.order_depths[self.product].buy_orders.items())
    
    @property
    def asks(self) -> list[tuple[int, int]]:
        """Return ask orders.

        Returns:
            dict[int, int].items(): ask orders (prc, amt)

        """
        return list(self._state.order_depths[self.product].sell_orders.items())
    
    @classmethod
    def _cls_update_bids(cls, product, prc, new_amt):
        if new_amt > 0:
            cls._state.order_depths[product].buy_orders[prc] = new_amt
        elif new_amt == 0:
            cls._state.order_depths[product].buy_orders[prc] = 0
        # else:
        #     raise ValueError("Negative amount in bid orders")

    @classmethod
    def _cls_update_asks(cls, product, prc, new_amt):
        if new_amt < 0:
            cls._state.order_depths[product].sell_orders[prc] = new_amt
        elif new_amt == 0:
            cls._state.order_depths[product].sell_orders[prc] = 0
        # else:
        #     raise ValueError("Positive amount in ask orders")
        
    def update_bids(self, prc: int, new_amt: int) -> None:
        """Update bid orders.

        Args:
            prc (int): price
            new_amt (int): new amount

        """
        self._cls_update_bids(self.product, prc, new_amt)
    
    def update_asks(self, prc: int, new_amt: int) -> None:
        """Update ask orders.

        Args:
            prc (int): price
            new_amt (int): new amount

        """
        self._cls_update_asks(self.product, prc, new_amt)

    @property
    def possible_buy_amt(self) -> int:
        """Return possible buy amount.

        Returns:
            int: possible buy amount
        
        """
        possible_buy_amount1 = self._position_limit[self.product] - self.rt_position
        possible_buy_amount2 = self._position_limit[self.product] - self.position
        return min(possible_buy_amount1, possible_buy_amount2)
        
    @property
    def possible_sell_amt(self) -> int:
        """Return possible sell amount.

        Returns:
            int: possible sell amount
        
        """
        possible_sell_amount1 = self._position_limit[self.product] + self.rt_position
        possible_sell_amount2 = self._position_limit[self.product] + self.position
        return min(possible_sell_amount1, possible_sell_amount2)

    def hist_mid_prc(self, size:int) -> np.ndarray:
        """Return historical mid price.

        Args:
            size (int): size of data

        Returns:
            np.ndarray: historical mid price
        
        """
        return (self.hist_order_depth('bidprc', 1, size) + self.hist_order_depth('askprc', 1, size)) / 2
    
    def hist_maxamt_askprc(self, size:int) -> np.ndarray:
        """Return price of ask order with maximum amount in historical order depth.

        Args:
            size (int): size of data

        Returns:
            int: price of ask order with maximum amount in historical order depth
        
        """
        res_array = np.empty(size)
        prc_array = np.array([self.hist_order_depth('askprc', 1, size), self.hist_order_depth('askprc', 2, size), self.hist_order_depth('askprc', 3, size)]).T
        amt_array = np.array([self.hist_order_depth('askamt', 1, size), self.hist_order_depth('askamt', 2, size), self.hist_order_depth('askamt', 3, size)]).T

        for i, amt_arr in enumerate(amt_array):
            res_array[i] = prc_array[i,np.nanargmax(amt_arr)]

        return res_array

    def hist_maxamt_bidprc(self, size:int) -> np.ndarray:
        """Return price of ask order with maximum amount in historical order depth.

        Args:
            size (int): size of data

        Returns:
            int: price of ask order with maximum amount in historical order depth
        
        """
        res_array = np.empty(size)
        prc_array = np.array([self.hist_order_depth('bidprc', 1, size), self.hist_order_depth('bidprc', 2, size), self.hist_order_depth('bidprc', 3, size)]).T
        amt_array = np.array([self.hist_order_depth('bidamt', 1, size), self.hist_order_depth('bidamt', 2, size), self.hist_order_depth('bidamt', 3, size)]).T

        for i, amt_arr in enumerate(amt_array):
            res_array[i] = prc_array[i,np.nanargmax(amt_arr)]

        return res_array
    
    def hist_vwap_all(self, size:int) -> np.ndarray:
        res_array = np.zeros(size)
        volsum_array = np.zeros(size)
        for i in range(1,4):
            tmp_bid_vol = self.hist_order_depth(f'bidamt', i, size)
            tmp_ask_vol = self.hist_order_depth(f'askamt', i, size)
            tmp_bid_prc = self.hist_order_depth(f'bidprc', i, size)
            tmp_ask_prc = self.hist_order_depth(f'askprc', i, size)
            if i == 0:
                res_array = res_array + (tmp_bid_prc*tmp_bid_vol) + (-tmp_ask_prc*tmp_ask_vol)
                volsum_array = volsum_array + tmp_bid_vol - tmp_ask_vol
            else:
                bid_nan_idx = np.isnan(tmp_bid_prc)
                ask_nan_idx = np.isnan(tmp_ask_prc)
                res_array = res_array + np.where(bid_nan_idx, 0, tmp_bid_prc*tmp_bid_vol) + np.where(ask_nan_idx, 0, -tmp_ask_prc*tmp_ask_vol)
                volsum_array = volsum_array + np.where(bid_nan_idx, 0, tmp_bid_vol) - np.where(ask_nan_idx, 0, tmp_ask_vol)
                
        return res_array / volsum_array
    
    def hist_obs_humidity(self, size:int) -> np.ndarray:
        return np.array(self._hist_observation['humidity'][-size:], dtype=np.float32)
    
    def hist_obs_sunlight(self, size:int) -> np.ndarray:
        return np.array(self._hist_observation['sunlight'][-size:], dtype=np.float32)
    
    def hist_obs_transportFees(self, size:int) -> np.ndarray:
        return np.array(self._hist_observation['transportFees'][-size:], dtype=np.float32)
    
    def hist_obs_exportTariff(self, size:int) -> np.ndarray:
        return np.array(self._hist_observation['exportTariff'][-size:], dtype=np.float32)
    
    def hist_obs_importTariff(self, size:int) -> np.ndarray:
        return np.array(self._hist_observation['importTariff'][-size:], dtype=np.float32)
    
    def hist_obs_bidPrice(self, size:int) -> np.ndarray:
        return np.array(self._hist_observation['bidPrice'][-size:], dtype=np.float32)
    
    def hist_obs_askPrice(self, size:int) -> np.ndarray:
        return np.array(self._hist_observation['askPrice'][-size:], dtype=np.float32)

    @property
    def best_bid(self) -> int:
        """Return best bid price and amount.

        Returns:
            tuple[int, int]: (price, amount)
        
        """
        buy_orders = self._state.order_depths[self.product].buy_orders
        if len(buy_orders) > 0:
            return max(buy_orders.keys())
        else:
            return self.best_ask - 1

    @property
    def best_ask(self) -> int:
        sell_orders = self._state.order_depths[self.product].sell_orders
        if len(sell_orders) > 0:
            return min(sell_orders.keys())
        else:
            return self.best_bid + 1

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2
    
    @property
    def bid_ask_spread(self) -> int:
        return self.best_ask - self.best_bid

    @property
    def best_bid_amount(self) -> int:
        """Return best bid price and amount.

        Returns:
            tuple[int, int]: (price, amount)
        
        """
        best_prc = max(self._state.order_depths[self.product].buy_orders.keys())
        best_amt = self._state.order_depths[self.product].buy_orders[best_prc]
        return best_amt
        
    @property
    def best_ask_amount(self) -> int:
        """Return best ask price and amount.

        Returns:
            tuple[int, int]: (price, amount)
        
        """
        best_prc = min(self._state.order_depths[self.product].sell_orders.keys())
        best_amt = self._state.order_depths[self.product].sell_orders[best_prc]
        return -best_amt
    
    @property
    def worst_bid(self) -> int:
        buy_orders = self._state.order_depths[self.product].buy_orders
        if len(buy_orders) > 0:
            return min(buy_orders.keys())
        else:
            return self.best_ask - 1

    @property
    def worst_ask(self) -> int:
        sell_orders = self._state.order_depths[self.product].sell_orders
        if len(sell_orders) > 0:
            return max(sell_orders.keys())
        else:
            return self.best_bid + 1

    @property
    def vwap(self) -> float:
        vwap = 0
        total_amt = 0

        for prc, amt in self._state.order_depths[self.product].buy_orders.items():
            vwap += (prc * amt)
            total_amt += amt

        for prc, amt in self._state.order_depths[self.product].sell_orders.items():
            vwap += (prc * abs(amt))
            total_amt += abs(amt)

        vwap /= total_amt
        return vwap

    @property
    def vwap_bidprc(self) -> float:
        """Return volume weighted average price of bid orders.

        Returns:
            float: volume weighted average price of bid orders

        """
        vwap = 0
        for prc, amt in self._state.order_depths[self.product].buy_orders.items():
            vwap += (prc * amt)
        vwap /= sum(self._state.order_depths[self.product].buy_orders.values())
        return vwap
    
    @property
    def vwap_askprc(self) -> float:
        """Return volume weighted average price of ask orders.

        Returns:
            float: volume weighted average price of ask orders

        """
        vwap = 0
        for prc, amt in self._state.order_depths[self.product].sell_orders.items():
            vwap += (prc * -amt)
        vwap /= -sum(self._state.order_depths[self.product].sell_orders.values())
        return vwap

    @property
    def maxamt_bidprc(self) -> int:
        """Return price of bid order with maximum amount.
        
        Returns:
            int: price of bid order with maximum amount

        """
        prc_max_mat, max_amt = 0,0
        for prc, amt in self._state.order_depths[self.product].buy_orders.items():
            if amt > max_amt:
                max_amt = amt
                prc_max_mat = prc

        return prc_max_mat
    
    @property
    def maxamt_askprc(self) -> int:
        """Return price of ask order with maximum amount.

        Returns:
            int: price of ask order with maximum amount
        
        """
        prc_max_mat, max_amt = 0,0
        for prc, amt in self._state.order_depths[self.product].sell_orders.items():
            if amt < max_amt:
                max_amt = amt
                prc_max_mat = prc

        return prc_max_mat
    
    @property
    def maxamt_midprc(self) -> float:
        return (self.maxamt_bidprc + self.maxamt_askprc) / 2

    def bidamt(self, price) -> int:
        order_depth = self._state.order_depths[self.product].buy_orders
        if price in order_depth.keys():
            return order_depth[price]
        else:
            return 0
        
    def askamt(self, price) -> int:
        order_depth = self._state.order_depths[self.product].sell_orders
        if price in order_depth.keys():
            return order_depth[price]
        else:
            return 0

    @property
    def total_bidamt(self) -> int:
        return sum(self._state.order_depths[self.product].buy_orders.values())

    @property
    def total_askamt(self) -> int:
        return -sum(self._state.order_depths[self.product].sell_orders.values())

    @property
    def orchid_south_bidprc(self) -> float:
        return self._state.observations.conversionObservations[self.product].bidPrice
    
    @property
    def orchid_south_askprc(self) -> float:
        return self._state.observations.conversionObservations[self.product].askPrice
    
    @property
    def orchid_south_midprc(self) -> float:
        return (self.orchid_south_bidprc + self.orchid_south_askprc) / 2
    
    @property
    def stoarageFees(self) -> float:
        return 0.1
    
    @property
    def transportFees(self) -> float:
        return self._state.observations.conversionObservations[self.product].transportFees
    
    @property
    def exportTariff(self) -> float:
        return self._state.observations.conversionObservations[self.product].exportTariff
    
    @property
    def importTariff(self) -> float:
        return self._state.observations.conversionObservations[self.product].importTariff
    
    @property
    def sunlight(self) -> float:
        return self._state.observations.conversionObservations[self.product].sunlight
    
    @property
    def humidity(self) -> float:
        return self._state.observations.conversionObservations[self.product].humidity

    @property
    def market_trades(self) -> list:
        return self._state.market_trades.get(self.product, [])


def linear_regression(X, y):
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.linalg.inv(X_bias.T.dot(X_bias)).dot(X_bias.T).dot(y)
    return theta

def cal_tau(day, timestep, T=1):
    return T - ((day - 1) * 20000 + timestep) * 2e-7

def cal_call(S, tau, sigma=0.16, r=0, K=10000):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
    delta = normalDist.cdf(d1)
    d2 = d1 - sigma * np.sqrt(tau)
    call_price = S * delta - K * math.exp(-r * tau) * normalDist.cdf(d2)
    return call_price, delta

def cal_imvol(market_price, S, tau, r=0, K=10000, tol=1e-6, max_iter=100):
    sigma = 0.16
    diff = cal_call(S, tau, sigma)[0] - market_price

    iter_count = 0
    while np.any(np.abs(diff) > tol) and iter_count < max_iter:
        vega = (cal_call(S, tau, sigma+tol)[0] - cal_call(S, tau, sigma)[0]) / tol
        sigma -= diff / vega
        diff = cal_call(S, tau, sigma)[0] - market_price
        iter_count += 1
    
    return sigma


class ExecutionProb:

    @staticmethod
    def orchids(delta):
        if delta < -1:
            return 0.571
        elif delta > -0.5:
            return 0
        elif delta == -1.0:
            return 0.2685
        elif delta == -0.75:
            return 0.2107
        elif delta == -0.5:
            return 0.1699


class Strategy:

    @staticmethod
    def arb(state: Status, fair_price):
        orders = []

        for ask_price, ask_amount in state.asks:
            if ask_price < fair_price:
                buy_amount = min(-ask_amount, state.possible_buy_amt)
                if buy_amount > 0:
                    orders.append(Order(state.product, int(ask_price), int(buy_amount)))
                    state.rt_position_update(state.rt_position + buy_amount)
                    state.update_asks(ask_price, -(-ask_amount - buy_amount))

            elif ask_price == fair_price:
                if state.rt_position < 0:
                    buy_amount = min(-ask_amount, -state.rt_position)
                    orders.append(Order(state.product, int(ask_price), int(buy_amount)))
                    state.rt_position_update(state.rt_position + buy_amount)
                    state.update_asks(ask_price, -(-ask_amount - buy_amount))

        for bid_price, bid_amount in state.bids:
            if bid_price > fair_price:
                sell_amount = min(bid_amount, state.possible_sell_amt)
                if sell_amount > 0:
                    orders.append(Order(state.product, int(bid_price), -int(sell_amount)))
                    state.rt_position_update(state.rt_position - sell_amount)
                    state.update_bids(bid_price, bid_amount - sell_amount)

            elif bid_price == fair_price:
                if state.rt_position > 0:
                    sell_amount = min(bid_amount, state.rt_position)
                    orders.append(Order(state.product, int(bid_price), -int(sell_amount)))
                    state.rt_position_update(state.rt_position - sell_amount)
                    state.update_bids(bid_price, bid_amount - sell_amount)

        return orders

    @staticmethod
    def mm_glft(
        state: Status,
        fair_price,
        mu=0,
        sigma=0.3959,
        gamma=1e-9,
        order_amount=20,
    ):
        
        q = state.rt_position / order_amount
        #Q = state.position_limit / order_amount

        kappa_b = 1 / max((fair_price - state.best_bid) - 1, 1)
        kappa_a = 1 / max((state.best_ask - fair_price) - 1, 1)

        A_b = 0.25
        A_a = 0.25

        delta_b = 1 / gamma * math.log(1 + gamma / kappa_b) + (-mu / (gamma * sigma**2) + (2 * q + 1) / 2) * math.sqrt((sigma**2 * gamma) / (2 * kappa_b * A_b) * (1 + gamma / kappa_b)**(1 + kappa_b / gamma))
        delta_a = 1 / gamma * math.log(1 + gamma / kappa_a) + (mu / (gamma * sigma**2) - (2 * q - 1) / 2) * math.sqrt((sigma**2 * gamma) / (2 * kappa_a * A_a) * (1 + gamma / kappa_a)**(1 + kappa_a / gamma))

        p_b = round(fair_price - delta_b)
        p_a = round(fair_price + delta_a)

        p_b = min(p_b, fair_price) # Set the buy price to be no higher than the fair price to avoid losses
        p_b = min(p_b, state.best_bid + 1) # Place the buy order as close as possible to the best bid price
        p_b = max(p_b, state.maxamt_bidprc + 1) # No market order arrival beyond this price

        p_a = max(p_a, fair_price)
        p_a = max(p_a, state.best_ask - 1)
        p_a = min(p_a, state.maxamt_askprc - 1)

        buy_amount = min(order_amount, state.possible_buy_amt)
        sell_amount = min(order_amount, state.possible_sell_amt)

        orders = []
        if buy_amount > 0:
            orders.append(Order(state.product, int(p_b), int(buy_amount)))
        if sell_amount > 0:
            orders.append(Order(state.product, int(p_a), -int(sell_amount)))
        return orders

    @staticmethod
    def mm_ou(
        state: Status,
        fair_price,
        gamma=1e-9,
        order_amount=20,
    ):

        q = state.rt_position / order_amount
        Q = state.position_limit / order_amount

        kappa_b = 1 / max((fair_price - state.best_bid) - 1, 1)
        kappa_a = 1 / max((state.best_ask - fair_price) - 1, 1)
            
        vfucn = lambda q,Q: -INF if (q==Q+1 or q==-(Q+1)) else math.log(math.sin(((q+Q+1)*math.pi)/(2*Q+2)))

        delta_b = 1 / gamma * math.log(1 + gamma / kappa_b) - 1 / kappa_b * (vfucn(q + 1, Q) - vfucn(q, Q))
        delta_a = 1 / gamma * math.log(1 + gamma / kappa_a) + 1 / kappa_a * (vfucn(q, Q) - vfucn(q - 1, Q))

        p_b = round(fair_price - delta_b)
        p_a = round(fair_price + delta_a)

        p_b = min(p_b, fair_price) # Set the buy price to be no higher than the fair price to avoid losses
        p_b = min(p_b, state.best_bid + 1) # Place the buy order as close as possible to the best bid price
        p_b = max(p_b, state.maxamt_bidprc + 1) # No market order arrival beyond this price

        p_a = max(p_a, fair_price)
        p_a = max(p_a, state.best_ask - 1)
        p_a = min(p_a, state.maxamt_askprc - 1)

        buy_amount = min(order_amount, state.possible_buy_amt)
        sell_amount = min(order_amount, state.possible_sell_amt)

        orders = []
        if buy_amount > 0:
            orders.append(Order(state.product, int(p_b), int(buy_amount)))
        if sell_amount > 0:
            orders.append(Order(state.product, int(p_a), -int(sell_amount)))
        return orders

    @staticmethod
    def exchange_arb(state: Status, fair_price, next_price_move=0):
        cost = state.transportFees + state.importTariff

        my_ask = state.maxamt_bidprc
        ask_max_expected_profit = 0
        optimal_my_ask = INF
        while my_ask < fair_price:
            delta = my_ask - fair_price
            execution_prob = ExecutionProb.orchids(delta)

            if my_ask > state.best_bid:
                trading_profit = my_ask - (state.orchid_south_askprc + next_price_move)
                expected_profit = execution_prob * (trading_profit - cost)
             
            else:
                execution_prob_list = []
                price_list = []
                amount_list = []

                for price, amount in state.bids:
                    if price >= my_ask:
                        execution_prob_list.append(1)
                        price_list.append(price)
                        amount_list.append(amount)

                total_amount = np.sum(amount_list)
                if total_amount < state.position_limit:
                    execution_prob_list.append(ExecutionProb.orchids(delta))
                    price_list.append(my_ask)
                    amount_list.append(state.position_limit - total_amount)

                trading_profit_list = np.array(price_list) - (state.orchid_south_askprc + next_price_move)
                expected_profit = (np.array(execution_prob_list) * (np.array(trading_profit_list) - cost) * np.array(amount_list) / state.position_limit).sum()

            if expected_profit > ask_max_expected_profit:
                optimal_my_ask = my_ask
                ask_max_expected_profit = expected_profit
            
            my_ask += 1


        cost = state.transportFees + state.exportTariff + state.stoarageFees

        my_bid = state.maxamt_askprc
        bid_max_expected_profit = 0
        optimal_my_bid = 1
        while my_bid > fair_price:
            delta = fair_price - my_bid
            execution_prob = ExecutionProb.orchids(delta)

            if my_bid < state.best_ask:
                trading_profit = (state.orchid_south_bidprc + next_price_move) - my_bid
                expected_profit = execution_prob * (trading_profit - cost)
             
            else:
                execution_prob_list = []
                price_list = []
                amount_list = []

                for price, amount in state.asks:
                    if price <= my_bid:
                        execution_prob_list.append(1)
                        price_list.append(price)
                        amount_list.append(abs(amount))

                total_amount = np.sum(amount_list)
                if total_amount < state.position_limit:
                    execution_prob_list.append(ExecutionProb.orchids(delta))
                    price_list.append(my_bid)
                    amount_list.append(state.position_limit - total_amount)

                trading_profit_list = (state.orchid_south_bidprc + next_price_move) - np.array(price_list)
                expected_profit = (np.array(execution_prob_list) * (np.array(trading_profit_list) - cost) * np.array(amount_list) / state.position_limit).sum()

            if expected_profit > bid_max_expected_profit:
                optimal_my_bid = my_bid
                bid_max_expected_profit = expected_profit
            
            my_bid -= 1
            

        orders = []
        
        if ask_max_expected_profit >= bid_max_expected_profit and ask_max_expected_profit > 0:
            orders.append(Order(state.product, int(optimal_my_ask), -int(state.position_limit)))
        elif bid_max_expected_profit > ask_max_expected_profit and bid_max_expected_profit > 0:
            orders.append(Order(state.product, int(optimal_my_bid), int(state.position_limit)))
        
        return orders

    @staticmethod
    def convert(state: Status):
        if state.position < 0:
            return -state.position
        elif state.position > 0:
            return -state.position
        else:
            return 0
        
    @staticmethod
    def index_arb(
        basket: Status,
        chocolate: Status,
        strawberries: Status,
        roses: Status,
        theta=380,
        threshold=30,
    ):
        
        basket_prc = basket.mid
        underlying_prc = 4 * chocolate.vwap + 6 * strawberries.vwap + 1 * roses.vwap
        spread = basket_prc - underlying_prc
        norm_spread = spread - theta

        orders = []
        if norm_spread > threshold:
            orders.append(Order(basket.product, int(basket.worst_bid), -int(basket.possible_sell_amt)))
        elif norm_spread < -threshold:
            orders.append(Order(basket.product, int(basket.worst_ask), int(basket.possible_buy_amt)))

        return orders
    
    @staticmethod
    def vol_arb(option: Status, iv, hv=0.16, threshold=0.00178):

        vol_spread = iv - hv

        orders = []

        if vol_spread > threshold:
            sell_amount = option.possible_sell_amt
            orders.append(Order(option.product, option.worst_bid, -sell_amount))
            executed_amount = min(sell_amount, option.total_bidamt)
            option.rt_position_update(option.rt_position - executed_amount)

        elif vol_spread < -threshold:
            buy_amount = option.possible_buy_amt
            orders.append(Order(option.product, option.worst_ask, buy_amount))
            executed_amount = min(buy_amount, option.total_askamt)
            option.rt_position_update(option.rt_position + executed_amount)

        return orders
    
    @staticmethod
    def delta_hedge(underlying: Status, option: Status, delta, rebalance_threshold=30):

        target_position = -round(option.rt_position * delta)
        current_position = underlying.position
        position_diff = target_position - current_position

        orders = []

        if underlying.bid_ask_spread == 1 and abs(position_diff) > rebalance_threshold:

            if position_diff < 0:
                sell_amount = min(abs(position_diff), underlying.possible_sell_amt)
                orders.append(Order(underlying.product, underlying.best_bid, -sell_amount))

            elif position_diff > 0:
                buy_amount = min(position_diff, underlying.possible_buy_amt)
                orders.append(Order(underlying.product, underlying.best_ask, buy_amount))
        
        return orders
    
    @staticmethod
    def insider_trading(signal_product: Status, trade_product: Status):

        buy_timestamp, sell_timestamp = 0, 0

        for trade in signal_product.market_trades:
            if trade.buyer == "Rhianna":
                buy_timestamp = trade.timestamp
            elif trade.seller == "Rhianna":
                sell_timestamp = trade.timestamp

        orders = []
        if buy_timestamp > sell_timestamp:
            orders.append(Order(trade_product.product, trade_product.worst_ask, trade_product.possible_buy_amt))
        elif buy_timestamp < sell_timestamp:
            orders.append(Order(trade_product.product, trade_product.worst_bid, -trade_product.possible_sell_amt))

        return orders


class Trade:

    @staticmethod   
    def amethysts(state: Status) -> list[Order]:

        current_price = state.maxamt_midprc

        orders = []
        orders.extend(Strategy.arb(state=state, fair_price=current_price))
        orders.extend(Strategy.mm_ou(state=state, fair_price=current_price, gamma=0.1, order_amount=20))

        return orders
    
    @staticmethod
    def starfruit(state: Status) -> list[Order]:

        current_price = state.maxamt_midprc

        orders = []
        orders.extend(Strategy.arb(state=state, fair_price=current_price))
        orders.extend(Strategy.mm_glft(state=state, fair_price=current_price, gamma=0.1, order_amount=20))

        return orders
    
    @staticmethod
    def orchids(state: Status) -> list[Order]:

        current_price = state.orchid_south_midprc

        # humidity = state.hist_obs_humidity(2)
        # if len(humidity) == 2:
        #     production_penalty = np.minimum(humidity, 60) - 60
        #     production_penalty += 80 - np.maximum(humidity, 80)
        #     next_price_move = 2.281 * (production_penalty[-1] - production_penalty[-2])
        # else:
        #     next_price_move = 0
            
        orders = []
        orders.extend(Strategy.exchange_arb(state=state, fair_price=current_price, next_price_move=0))

        return orders
    
    @staticmethod
    def convert(state: Status) -> int:
        return Strategy.convert(state=state)
    
    @staticmethod
    def gift_basket(basket: Status, chocolate: Status, strawberries: Status, roses: Status) -> list[Order]:

        orders = []
        orders.extend(Strategy.index_arb(basket, chocolate, strawberries, roses, threshold=30))
        orders.extend(Strategy.informed_trading(roses, basket))

        return orders
    
    @staticmethod
    def roses(state: Status) -> list[Order]:

        orders = []
        orders.extend(Strategy.insider_trading(state, state))

        return orders

    @staticmethod
    def coconut(underlying: Status, option: Status, day) -> list[Order]:

        result = {
            option.product: [],
            underlying.product: []
        }

        underlying_prc = underlying.hist_mid_prc(1)[0]
        option_prc = option.hist_mid_prc(1)[0]

        tau = cal_tau(day=day, timestep=underlying.timestep)
        theo, delta = cal_call(underlying_prc, tau)
        iv = cal_imvol(option_prc, underlying_prc, tau)
        logger.print(f'{theo}, {delta}, {iv}')
        
        result[option.product].extend(Strategy.vol_arb(option, iv, threshold=0.00175))
        result[underlying.product].extend(Strategy.delta_hedge(underlying, option, delta, rebalance_threshold=60))

        return result
    

class Trader:

    state_amethysts = Status('AMETHYSTS')
    state_starfruit = Status('STARFRUIT')
    state_orchids = Status('ORCHIDS')
    state_chocolate = Status('CHOCOLATE')
    state_strawberries = Status('STRAWBERRIES')
    state_roses = Status('ROSES')
    state_gift_basket = Status('GIFT_BASKET')
    state_coconut = Status('COCONUT')
    state_coconut_coupon = Status('COCONUT_COUPON')

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        Status.cls_update(state)

        result = {}

        # round 1
        result["AMETHYSTS"] = Trade.amethysts(self.state_amethysts)
        result["STARFRUIT"] = Trade.starfruit(self.state_starfruit)
        
        # round 2
        result["ORCHIDS"] = Trade.orchids(self.state_orchids)
        conversions = Trade.convert(self.state_orchids)

        # round 3
        result["GIFT_BASKET"] = Trade.gift_basket(self.state_gift_basket, self.state_chocolate, self.state_strawberries, self.state_roses)
        result["ROSES"] = Trade.roses(self.state_roses)
        
        # round 4
        coconut_result = Trade.coconut(self.state_coconut, self.state_coconut_coupon, day=5)
        result["COCONUT_COUPON"] = coconut_result["COCONUT_COUPON"]
        # result["COCONUT"] = coconut_result["COCONUT"]

        traderData = "SAMPLE" 
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
