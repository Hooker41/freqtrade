# pragma pylint: disable=missing-docstring, W0212, too-many-arguments

"""
This module contains the backtesting logic
"""
import logging
import operator
from argparse import Namespace
from typing import Dict, Tuple, Any, List, Optional

import arrow
from pandas import DataFrame
from tabulate import tabulate

import freqtrade.optimize as optimize
from freqtrade import exchange
from freqtrade.analyze import Analyze
from freqtrade.arguments import Arguments
from freqtrade.configuration import Configuration
from freqtrade.exchange import Bittrex
from freqtrade.misc import file_dump_json
from freqtrade.persistence import Trade
from pandas import DataFrame, to_datetime
import pandas as pd
from freqtrade.optimize import ml_utils
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


class Backtesting(object):
    """
    Backtesting class, this class contains all the logic to run a backtest

    To run a backtest:
    backtesting = Backtesting(config)
    backtesting.start()
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.analyze = None
        self.ticker_interval = None
        self.ML_tickerdata_to_dataframe = None
        self.populate_buy_trend = None
        self.populate_sell_trend = None
        self.slippage = None
        self._init()

    def _init(self) -> None:
        """
        Init objects required for backtesting
        :return: None
        """
        self.analyze = Analyze(self.config)
        self.ticker_interval = self.analyze.strategy.ticker_interval
        self.slippage = self.analyze.strategy.slippage
        self.ML_tickerdata_to_dataframe = self.analyze.ML_tickerdata_to_dataframe
        self.populate_buy_trend = self.analyze.populate_buy_trend
        self.populate_sell_trend = self.analyze.populate_sell_trend
        exchange._API = Bittrex({'key': '', 'secret': ''})

    @staticmethod
    def get_timeframe(data: Dict[str, DataFrame]) -> Tuple[arrow.Arrow, arrow.Arrow]:
        """
        Get the maximum timeframe for the given backtest data
        :param data: dictionary with preprocessed backtesting data
        :return: tuple containing min_date, max_date
        """
        timeframe = [
            (arrow.get(min(frame.date)), arrow.get(max(frame.date)))
            for frame in data.values()
        ]
        return min(timeframe, key=operator.itemgetter(0))[0], \
            max(timeframe, key=operator.itemgetter(1))[1]

    def _generate_text_table(self, data: Dict[str, Dict], results: DataFrame) -> str:
        """
        Generates and returns a text table for the given backtest data and the results dataframe
        :return: pretty printed table with tabulate as str
        """
        stake_currency = self.config.get('stake_currency')

        floatfmt = ('s', 'd', '.2f', '.8f', '.1f')
        tabular_data = []
        headers = ['pair', 'buy count', 'avg profit %',
                   'total profit ' + stake_currency, 'avg duration', 'profit', 'loss']
        for pair in data:
            result = results[results.currency == pair]
            tabular_data.append([
                pair,
                len(result.index),
                result.profit_percent.mean() * 100.0,
                result.profit_BTC.sum(),
                result.duration.mean(),
                len(result[result.profit_BTC > 0]),
                len(result[result.profit_BTC < 0])
            ])

        # Append Total
        tabular_data.append([
            'TOTAL',
            len(results.index),
            results.profit_percent.mean() * 100.0,
            results.profit_BTC.sum(),
            results.duration.mean(),
            len(results[results.profit_BTC > 0]),
            len(results[results.profit_BTC < 0])
        ])
        return tabulate(tabular_data, headers=headers, floatfmt=floatfmt)

    def _get_sell_trade_entry(
            self, pair: str, buy_row: DataFrame,
            partial_ticker: List, trade_count_lock: Dict, args: Dict) -> Optional[Tuple]:

        stake_amount = args['stake_amount']
        max_open_trades = args.get('max_open_trades', 0)
        trade = Trade(
            open_rate=buy_row.close + self.slippage,             #implement slippage 0.01 for buy_row
            open_date=buy_row.date,
            stake_amount=stake_amount,
            amount=stake_amount / buy_row.open,
            fee=exchange.get_fee()
        )

        # calculate win/lose forwards from buy point
        for sell_row in partial_ticker:
            if max_open_trades > 0:
                # Increase trade_count_lock for every iteration
                trade_count_lock[sell_row.date] = trade_count_lock.get(sell_row.date, 0) + 1

            buy_signal = sell_row.buy
            #implement slippage 0.01 for sell_row

            if self.analyze.should_sell(trade, sell_row.close - self.slippage, sell_row.date, buy_signal,
                                        sell_row.sell):
                return \
                    sell_row, \
                    (
                        pair,
                        trade.calc_profit_percent(rate=sell_row.close),
                        trade.calc_profit(rate=sell_row.close),
                        (sell_row.date - buy_row.date).seconds // 60
                    ), \
                    sell_row.date
        return None

    def backtest(self, args: Dict) -> DataFrame:
        """
        Implements backtesting functionality

        NOTE: This method is used by Hyperopt at each iteration. Please keep it optimized.
        Of course try to not have ugly code. By some accessor are sometime slower than functions.
        Avoid, logging on this method

        :param args: a dict containing:
            stake_amount: btc amount to use for each trade
            processed: a processed dictionary with format {pair, data}
            max_open_trades: maximum number of concurrent trades (default: 0, disabled)
            realistic: do we try to simulate realistic trades? (default: True)
            sell_profit_only: sell if profit only
            use_sell_signal: act on sell-signal
        :return: DataFrame
        """
        headers = ['date', 'buy', 'open', 'close', 'sell']
        processed = args['processed']
        max_open_trades = args.get('max_open_trades', 0)
        realistic = args.get('realistic', False)
        record = args.get('record', None)
        records = []
        trades = []
        trade_count_lock = {}
        for pair, pair_data in processed.items():
            pair_data['buy'], pair_data['sell'] = 0, 0  # cleanup from previous run

            ticker_data = self.populate_sell_trend(self.populate_buy_trend(pair_data))[headers]
            ticker = [x for x in ticker_data.itertuples()]

            lock_pair_until = None
            for index, row in enumerate(ticker):
                if row.buy == 0 or row.sell == 1:
                    continue  # skip rows where no buy signal or that would immediately sell off

                if realistic:
                    if lock_pair_until is not None and row.date <= lock_pair_until:
                        continue
                if max_open_trades > 0:
                    # Check if max_open_trades has already been reached for the given date
                    if not trade_count_lock.get(row.date, 0) < max_open_trades:
                        continue

                    trade_count_lock[row.date] = trade_count_lock.get(row.date, 0) + 1

                ret = self._get_sell_trade_entry(pair, row, ticker[index + 1:],
                                                 trade_count_lock, args)

                if ret:
                    row2, trade_entry, next_date = ret
                    lock_pair_until = next_date
                    trades.append(trade_entry)
                    if record:
                        # Note, need to be json.dump friendly
                        # record a tuple of pair, current_profit_percent,
                        # entry-date, duration
                        records.append((pair, trade_entry[1],
                                        row.date.strftime('%s'),
                                        row2.date.strftime('%s'),
                                        index, trade_entry[3]))
        # For now export inside backtest(), maybe change so that backtest()
        # returns a tuple like: (dataframe, records, logs, etc)
        if record and record.find('trades') >= 0:
            logger.info('Dumping backtest results')
            file_dump_json('backtest-result.json', records)
        labels = ['currency', 'profit_percent', 'profit_BTC', 'duration']
        return DataFrame.from_records(trades, columns=labels)

    def run_buy_sell_strategy(self, data, initial_cash, transaction_cost, alpha):
        """Runs a simple strategy.

            Buys alpha*portfolio value when predicted up and sells alpha*portfolio
            value when predicted down.

            Assume that data has the following format
            [BTC-USD_Low
            BTC-USD_High
            BTC-USD_Open
            BTC-USD_Close
            BTC-USD_Volume
            BTC-USD_Prediction]

            The trading startegy takes 'Prediction' column to make buy/sell decision.
        """
        trading_portfolio = pd.DataFrame(index = data.index)
        trading_portfolio['Portfolio_value'] = pd.Series(index = data.index)
        trading_portfolio['Coins_held'] = pd.Series(index = data.index)
        trading_portfolio['Cash_held'] = pd.Series(index = data.index)

        coins_held = 0
        cash_held = initial_cash
        portfolio_value = coins_held + cash_held
        for i, row in enumerate(data.values):
            date = data.index[i]
            low, high, open, close, volume, prediction = row
            if prediction == 1:
                # Buy coin.
                transaction = min(alpha * portfolio_value, cash_held)
                fee = transaction * transaction_cost
                coins_held += transaction / close
                cash_held -= transaction - fee
            if prediction == 0:
                # Sell coin.
                transaction = min(alpha*portfolio_value,coins_held * close)
                fee = transaction * transaction_cost
                coins_held -= transaction / close
                cash_held += transaction - fee
            portfolio_value = coins_held*close + cash_held
            trading_portfolio['Portfolio_value'][date] = portfolio_value
            trading_portfolio['Coins_held'][date] = coins_held
            trading_portfolio['Cash_held'][date] = cash_held
        profit = portfolio_value - initial_cash
        ret = 100 * profit / initial_cash
        avg_value = trading_portfolio['Portfolio_value'].mean()
        std_value = trading_portfolio['Portfolio_value'].std()
        trading_stats = { 'profit_dollar': profit,
        'return_percent': ret,
        'avg_value': avg_value,
        'std value': std_value }
        return(trading_portfolio, trading_stats)

    def start(self) -> None:
        """
        Run a backtesting end-to-end
        :return: None
        """
        data = {}
        pairs = self.config['exchange']['pair_whitelist']
        logger.info('Using stake_currency: %s ...', self.config['stake_currency'])
        logger.info('Using stake_amount: %s ...', self.config['stake_amount'])

        timerange = Arguments.parse_timerange(self.config.get('timerange'))

        data = optimize.load_data(
            self.config['datadir'],
            pairs=pairs,
            ticker_interval=self.ticker_interval,
            refresh_pairs=False,
            timerange=timerange
        )

        preprocessed = self.ML_tickerdata_to_dataframe(data)

        out_coin = 'BTC_XMR'
        test_ratio = 0.2

        data = ml_utils.run_pipeline(preprocessed[out_coin], out_coin, test_ratio)
       
        initial_cash = 100 # in USD
        transaction_cost = 0.01 # as ratio for fee
        alpha = 0.05 # ratio of portfolio value that we trade each transaction
        (trading_portfolio, trading_stats) = self.run_buy_sell_strategy(data, initial_cash, transaction_cost, alpha)
        print(trading_portfolio)
        #print(trading_portfolio.head(),"Head of trading series.")
        print ("Stats from trading: ", trading_stats)
        # plots trading portfolio value in time next to out_coin price
        trading_portfolio.plot(subplots=True, figsize=(6, 6)); plt.legend(loc='best')
        plt.show()
        

def setup_configuration(args: Namespace) -> Dict[str, Any]:
    """
    Prepare the configuration for the backtesting
    :param args: Cli args from Arguments()
    :return: Configuration
    """
    configuration = Configuration(args)
    config = configuration.get_config()

    # Ensure we do not use Exchange credentials
    config['exchange']['key'] = ''
    config['exchange']['secret'] = ''

    return config


def start(args: Namespace) -> None:
    """
    Start Backtesting script
    :param args: Cli args from Arguments()
    :return: None
    """
    print(args)
    # Initialize configuration
    config = setup_configuration(args)
    logger.info('Starting freqtrade in Backtesting mode')

    # Initialize backtesting object
    backtesting = Backtesting(config)
    backtesting.start()
