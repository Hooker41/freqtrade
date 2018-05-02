# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.indicator_helpers import fishers_inverse
from freqtrade.strategy.interface import IStrategy


class MLStrategy(IStrategy):
    """
    Default Strategy provided by freqtrade bot.
    You can override it with your own strategy
    """

    # Minimal ROI designed for the strategy
    minimal_roi = {
        "40":  0.0,         #in 40min
        "30":  0.01,        #in 30min
        "20":  0.02,        #in 20min
        "0":  0.04
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.10

    # Optimal ticker interval for the strategy
    ticker_interval = 5

    # Slippage 
    slippage = 0.01
    
    @staticmethod
    def ML_parse_ticker_dataframe(pair: str, ticker: list) -> DataFrame:
        """
        Analyses the trend for the given ticker history
        :param ticker: See exchange.get_ticker_history
        :return: DataFrame
        """
        columns = {'C': pair+'_close', 'V': pair+'_volume', 'O': pair+'_open', 'H': pair+'_high', 'L': pair+'_low', 'T': pair+'_date'}
        frame = DataFrame(ticker).rename(columns=columns).set_index(pair+'_date')
        frame.index.names = [None]
        
        if 'BV' in frame:
            frame.drop('BV', axis=1, inplace=True)

        frame.index = to_datetime(frame.index, utc=True, infer_datetime_format=True)
        
        return frame
        
    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """
        print('ML_STRATEGY Dataframe', dataframe)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Chart type
        # ------------------------------------
        # Heikinashi stategy
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[ (dataframe['rsi'] < 35), 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        
        dataframe.loc[ (qtpylib.crossed_above(dataframe['rsi'], 70)), 'sell'] = 1
        return dataframe
