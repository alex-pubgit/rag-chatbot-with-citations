# Slippage

Sample data: AcmeSwap is a fictional product, used to demonstrate this chatbot.

## What slippage is

Slippage is the difference between the quoted price and the price at the moment your swap is executed. Prices move while a transaction waits to be confirmed.

## Slippage tolerance

The default slippage tolerance is 0.5%. You can adjust it in the swap settings, up to a maximum of 5%. If the price moves by more than your tolerance, the swap fails and no tokens are exchanged, but the network fee for the failed attempt is still spent.

## Choosing a value

A higher tolerance makes a swap less likely to fail and more likely to complete at a worse price. Raise it only for tokens with thin liquidity.
